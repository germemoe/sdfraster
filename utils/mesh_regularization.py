"""Mesh regularization module for normal consistency loss.

Based on MILo's mesh regularization approach.
Computes normal consistency loss between:
- Normals from Gaussian-rendered depth (via finite differences)
- Normals from mesh extracted via Marching Tetrahedra

Reference: MILo project regularization/regularizer/mesh.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tetmesh import marching_tetrahedra
from utils.mesh import Meshes, MeshRasterizer, MeshRenderer, transform_points_world_to_view
from utils.geometry_utils import depth_to_normal, transform_normal_world_to_view


class MeshRegularizer(nn.Module):
    """Mesh regularization for normal consistency.

    Extracts triangle mesh from tetrahedral model using Marching Tetrahedra,
    renders the mesh normals, and computes consistency loss with Gaussian normals.
    """

    def __init__(
        self,
        start_iter: int = 8001,
        stop_iter: int = 18000,
        normal_weight: float = 0.05,
        sdf_threshold: float = 0.0,
        use_opengl: bool = True,
    ):
        """Initialize mesh regularizer.

        Args:
            start_iter: Iteration to start mesh regularization
            stop_iter: Iteration to stop mesh regularization
            normal_weight: Weight for normal consistency loss
            sdf_threshold: SDF threshold for mesh extraction (default 0.0 = surface)
            use_opengl: Use OpenGL context for rasterization (vs CUDA)
        """
        super().__init__()
        self.start_iter = start_iter
        self.stop_iter = stop_iter
        self.normal_weight = normal_weight
        self.sdf_threshold = sdf_threshold
        self.use_opengl = use_opengl

        self._rasterizer = None
        self._renderer = None
        self._cached_mesh = None
        self._last_camera_size = None

    def should_apply(self, iteration: int) -> bool:
        """Check if regularization should be applied at this iteration."""
        return self.start_iter <= iteration < self.stop_iter

    def _get_renderer(self, camera):
        """Get or create mesh renderer for given camera."""
        current_size = (camera.image_height, camera.image_width)

        if self._rasterizer is None or self._last_camera_size != current_size:
            self._rasterizer = MeshRasterizer(
                cameras=[camera],
                use_opengl=self.use_opengl
            )
            self._renderer = MeshRenderer(self._rasterizer)
            self._last_camera_size = current_size

        return self._renderer

    def extract_mesh(self, model, threshold: float = None):
        """Extract triangle mesh from model using Marching Tetrahedra.

        Args:
            model: TetModel with vertices, indices, and query_vert_sdf method
            threshold: SDF threshold (uses self.sdf_threshold if None)

        Returns:
            Meshes: Extracted triangle mesh, or None if no surface found
        """
        if threshold is None:
            threshold = self.sdf_threshold

        # Get vertices and tetrahedra indices
        vertices = model.vertices.detach()  # (N_verts, 3)

        # 只使用活跃的四面体（排除 empty_indices）
        tets = model.indices  # (N_tets, 4)
        if hasattr(model, 'empty_indices') and model.empty_indices is not None and model.empty_indices.shape[0] > 0:
            # indices 已经是活跃的四面体，empty_indices 是被裁剪的
            # 确保我们只用 indices
            pass  # model.indices 应该已经是正确的

        # Query SDF values at all vertices
        vert_sdf = model.query_vert_sdf().detach()  # (N_verts,)

        # Shift SDF by threshold
        sdf_shifted = vert_sdf - threshold

        # Run marching tetrahedra
        mesh_verts, mesh_faces = marching_tetrahedra(vertices, tets, sdf_shifted)

        if mesh_verts.shape[0] == 0:
            return None

        return Meshes(verts=mesh_verts, faces=mesh_faces)

    def render_mesh_normal(self, mesh, camera):
        """Render mesh and get face normals in view space.

        Args:
            mesh: Meshes object
            camera: Camera object

        Returns:
            normal_map: (H, W, 3) normal map in view space
            valid_mask: (H, W) boolean mask for valid pixels
            depth_map: (H, W) mesh depth map
        """
        renderer = self._get_renderer(camera)

        # Render mesh (同时渲染 depth 和 normals，与 MILo 保持一致)
        output = renderer(
            mesh,
            cameras=[camera],
            cam_idx=0,
            return_depth=True,
            return_normals=True,
            return_pix_to_face=True,
        )

        # Get normals (in world space from face_normals)
        normals_world = output['normals']  # (1, H, W, 3)
        pix_to_face = output['pix_to_face']  # (1, H, W, 1)
        mesh_depth = output['depth'].squeeze()  # (H, W)

        # Valid mask where mesh is visible (depth > 0)
        valid_mask = (mesh_depth > 0)  # (H, W)

        # Transform normals from world to view space
        normals_view = transform_normal_world_to_view(
            normals_world.squeeze(0),  # (H, W, 3)
            camera
        )

        return normals_view, valid_mask

    def compute_gaussian_normal(self, depth_map, camera):
        """Compute normal from Gaussian-rendered depth map.

        Args:
            depth_map: (H, W) depth values from Gaussian rendering
            camera: Camera object

        Returns:
            normal_map: (3, H, W) normal map in view space
        """
        return depth_to_normal(camera, depth_map)

    def compute_normal_consistency_loss(
        self,
        mesh_normal: torch.Tensor,
        gaussian_normal: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute normal consistency loss.

        Loss = 1 - |dot(n_mesh, n_gaussian)|

        Using absolute value makes the loss invariant to normal direction flips.

        Args:
            mesh_normal: (H, W, 3) mesh normals in view space
            gaussian_normal: (3, H, W) or (H, W, 3) Gaussian normals in view space
            mask: Optional (H, W) boolean mask for valid pixels

        Returns:
            loss: Scalar normal consistency loss
        """
        # Ensure shape is (H, W, 3)
        if gaussian_normal.shape[0] == 3:
            gaussian_normal = gaussian_normal.permute(1, 2, 0)

        # Normalize both normals
        mesh_normal = F.normalize(mesh_normal, dim=-1)
        gaussian_normal = F.normalize(gaussian_normal, dim=-1)

        # Compute dot product
        dot_product = (mesh_normal * gaussian_normal).sum(dim=-1)  # (H, W)

        # Loss: 1 - |cos(angle)|
        loss = 1.0 - dot_product.abs()

        if mask is not None:
            # Apply mask
            loss = loss * mask.float()
            loss = loss.sum() / (mask.float().sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def forward(
        self,
        model,
        camera,
        gaussian_depth: torch.Tensor,
        iteration: int = None,
        gt_mask: torch.Tensor = None,  # GT alpha mask，排除背景散乱法线
    ) -> dict:
        """Compute mesh regularization loss.

        Args:
            model: TetModel with vertices, indices, query_vert_sdf
            camera: Camera object for rendering
            gaussian_depth: (H, W) depth from Gaussian rendering
            iteration: Current training iteration (for scheduling)
            gt_mask: (H, W) or (1, H, W) GT alpha mask，背景为0前景为1

        Returns:
            dict with:
                - 'loss': Weighted normal consistency loss (0 if not active)
                - 'normal_loss_raw': Unweighted loss (for logging)
                - 'mesh_normal': (H, W, 3) mesh normal map
                - 'gaussian_normal': (3, H, W) Gaussian normal map
                - 'active': Whether regularization was applied
        """
        result = {
            'loss': torch.tensor(0.0, device=gaussian_depth.device),
            'normal_loss_raw': torch.tensor(0.0, device=gaussian_depth.device),
            'mesh_normal': None,
            'gaussian_normal': None,
            'active': False,
        }

        # Check if regularization should be applied
        if iteration is not None and not self.should_apply(iteration):
            return result

        # Extract mesh from model
        mesh = self.extract_mesh(model)
        if mesh is None:
            print("[MeshRegularizer] Warning: No mesh extracted")
            return result

        # Render mesh normals
        mesh_normal, rasterization_mask = self.render_mesh_normal(mesh, camera)

        # Compute Gaussian normals from depth
        gaussian_normal = self.compute_gaussian_normal(gaussian_depth, camera)

        # Build combined mask
        H, W = gaussian_depth.shape

        # 1. Mesh rasterization mask (mesh渲染到的位置)
        combined_mask = rasterization_mask

        # 2. GT alpha mask (排除背景散乱法线)
        if gt_mask is not None:
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.squeeze(0)
            gt_mask_bool = gt_mask > 0.5
            combined_mask = combined_mask & gt_mask_bool

        # Compute normal consistency loss
        normal_loss = self.compute_normal_consistency_loss(
            mesh_normal, gaussian_normal, combined_mask
        )

        result['loss'] = self.normal_weight * normal_loss
        result['normal_loss_raw'] = normal_loss
        result['mesh_normal'] = mesh_normal
        result['gaussian_normal'] = gaussian_normal
        result['active'] = True

        return result


def create_mesh_regularizer(config: dict = None) -> MeshRegularizer:
    """Factory function to create mesh regularizer from config.

    Args:
        config: Dictionary with keys:
            - start_iter: int (default 8001)
            - stop_iter: int (default 18000)
            - normal_weight: float (default 0.05)
            - sdf_threshold: float (default 0.0)
            - use_opengl: bool (default True)

    Returns:
        MeshRegularizer instance
    """
    if config is None:
        config = {}

    return MeshRegularizer(
        start_iter=config.get('start_iter', 8001),
        stop_iter=config.get('stop_iter', 18000),
        normal_weight=config.get('normal_weight', 0.05),
        sdf_threshold=config.get('sdf_threshold', 0.0),
        use_opengl=config.get('use_opengl', True),
    )
