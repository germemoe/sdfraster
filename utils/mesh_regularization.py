"""Mesh regularization module for normal/depth consistency and occupancy loss.

Based on MILo's mesh regularization approach.
Computes:
- Normal consistency loss between mesh and Gaussian normals
- Depth consistency loss between mesh and Gaussian depths
- Surface occupancy loss (SDF at surface points should be ~0)

Reference: MILo project regularization/regularizer/mesh.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tetmesh import marching_tetrahedra
from utils.mesh import Meshes, MeshRasterizer, MeshRenderer, transform_points_world_to_view, evaluate_vertex_occupancy
from utils.geometry_utils import depth_to_normal, transform_normal_world_to_view


def depth_to_points_world(depth_map, camera):
    """Convert depth map to 3D points in world space.

    Args:
        depth_map: (H, W) depth values
        camera: Camera object

    Returns:
        points: (N, 3) 3D points in world space (only valid depth pixels)
        valid_mask: (H, W) boolean mask of valid pixels
    """
    device = depth_map.device
    H, W = depth_map.shape

    # Valid depth mask
    valid_mask = depth_map > 0

    # Compute focal lengths from FOV
    fx = W / (2 * math.tan(camera.fovx / 2.))
    fy = H / (2 * math.tan(camera.fovy / 2.))
    cx, cy = W / 2., H / 2.

    # Create pixel grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Pixel coordinates (centered)
    x = (grid_x + 0.5 - cx) / fx
    y = (grid_y + 0.5 - cy) / fy

    # 3D points in view space (depth is along z-axis)
    z = depth_map
    points_view = torch.stack([x * z, y * z, z], dim=-1)  # (H, W, 3)

    # Transform to world space
    # world_view_transform: world -> view, so we need inverse
    view_to_world = torch.inverse(camera.world_view_transform.to(device))  # (4, 4)
    R = view_to_world[:3, :3]
    t = view_to_world[:3, 3]

    points_world = points_view @ R.T + t  # (H, W, 3)

    # Extract valid points
    valid_points = points_world[valid_mask]  # (N, 3)

    return valid_points, valid_mask


class MeshRegularizer(nn.Module):
    """Mesh regularization for normal/depth consistency and surface occupancy.

    Extracts triangle mesh from tetrahedral model using Marching Tetrahedra,
    renders the mesh normals/depth, and computes consistency losses with Gaussian rendering.
    Also computes surface occupancy loss to ensure SDF is near 0 at surface points.
    """

    def __init__(
        self,
        start_iter: int = 8001,
        stop_iter: int = 18000,
        normal_weight: float = 0.05,
        depth_weight: float = 0.0,
        occupancy_weight: float = 0.0,
        occupancy_labels_weight: float = 0.0,  # [NEW] occupancy labels loss weight
        occupancy_labels_reset_interval: int = 1000,  # [NEW] how often to reset labels
        depth_scale: float = 0.1,
        depth_ratio: float = 0.5,  # MILo: ratio of median depth (not expected!)
        sdf_threshold: float = 0.0,
        use_opengl: bool = True,
    ):
        """Initialize mesh regularizer.

        Args:
            start_iter: Iteration to start mesh regularization
            stop_iter: Iteration to stop mesh regularization
            normal_weight: Weight for normal consistency loss
            depth_weight: Weight for depth consistency loss
            occupancy_weight: Weight for surface occupancy loss
            occupancy_labels_weight: Weight for occupancy labels loss (MILo-style)
            occupancy_labels_reset_interval: How often to recompute occupancy labels
            depth_scale: Scale factor for depth loss (controls sensitivity)
                         If None, will use model.scene_scaling in forward()
            depth_ratio: Ratio for median depth in mixing (MILo convention)
                         Final depth = (1 - depth_ratio) * expected + depth_ratio * median
                         Set to 0.0 to use only expected depth
            sdf_threshold: SDF threshold for mesh extraction (default 0.0 = surface)
            use_opengl: Use OpenGL context for rasterization (vs CUDA)
        """
        super().__init__()
        self.start_iter = start_iter
        self.stop_iter = stop_iter
        self.normal_weight = normal_weight
        self.depth_weight = depth_weight
        self.occupancy_weight = occupancy_weight
        self.occupancy_labels_weight = occupancy_labels_weight
        self.occupancy_labels_reset_interval = occupancy_labels_reset_interval
        self.depth_scale = depth_scale
        self.depth_ratio = depth_ratio
        self.sdf_threshold = sdf_threshold
        self.use_opengl = use_opengl

        self._rasterizer = None
        self._renderer = None
        self._cached_mesh = None
        self._last_camera_size = None

        # [NEW] Occupancy labels state
        self._cached_occupancy_labels = None
        self._occupancy_labels_last_reset_iter = -1
        self._occupancy_labels_num_verts = 0

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
        # MILo: 不做 detach，让梯度流回到 SDF 参数
        vertices = model.vertices  # (N_verts, 3)

        # 只使用活跃的四面体（排除 empty_indices）
        tets = model.indices  # (N_tets, 4)
        if hasattr(model, 'empty_indices') and model.empty_indices is not None and model.empty_indices.shape[0] > 0:
            # indices 已经是活跃的四面体，empty_indices 是被裁剪的
            # 确保我们只用 indices
            pass  # model.indices 应该已经是正确的

        # Query SDF values at all vertices
        # MILo: 不做 detach，让梯度流回到 SDF 参数
        vert_sdf = model.query_vert_sdf()  # (N_verts,)

        # Shift SDF by threshold
        sdf_shifted = vert_sdf - threshold

        # Run marching tetrahedra
        # 排除包含外部球面采样点的四面体
        num_internal = model.num_int_verts if hasattr(model, 'num_int_verts') else None
        mesh_verts, mesh_faces = marching_tetrahedra(vertices, tets, sdf_shifted, num_internal=num_internal)

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

        return normals_view, valid_mask, mesh_depth

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
        """Compute normal consistency loss (MILo mesh_normal_loss style).

        Loss = 1 - |dot(n_mesh, n_gaussian)|

        MILo uses abs() for mesh_normal_loss because:
        - Delaunay triangulation may not update frequently
        - Mesh could self-intersect and have flipped normals
        - We only care about plane alignment, not normal direction

        Args:
            mesh_normal: (H, W, 3) mesh normals in view space (already normalized from face_normals)
            gaussian_normal: (3, H, W) or (H, W, 3) Gaussian normals in view space (already normalized from render)
            mask: Optional (H, W) boolean mask for valid pixels

        Returns:
            loss: Scalar normal consistency loss
        """
        # Ensure shape is (H, W, 3)
        if gaussian_normal.shape[0] == 3:
            gaussian_normal = gaussian_normal.permute(1, 2, 0)

        # MILo 风格：不在 loss 计算时做 normalize
        # mesh_normal 已在 face_normals 属性中归一化
        # gaussian_normal 已在 render 函数中归一化
        # 直接计算点积

        # Compute dot product
        dot_product = (mesh_normal * gaussian_normal).sum(dim=-1)  # (H, W)

        # Loss: 1 - |cos(angle)| (MILo mesh_normal_loss uses abs)
        loss = 1.0 - dot_product.abs()

        if mask is not None:
            # Apply mask
            loss = loss * mask.float()
            loss = loss.sum() / (mask.float().sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def compute_depth_loss(
        self,
        mesh_depth: torch.Tensor,
        gaussian_depth: torch.Tensor,
        gaussian_depth_med: torch.Tensor = None,
        mask: torch.Tensor = None,
        scale: float = None,
        depth_ratio: float = None
    ) -> torch.Tensor:
        """Compute depth consistency loss (MILo-style).

        Loss = log(1 + |mesh_depth - mixed_gaussian_depth| / scale)

        Uses log to suppress large outliers (similar to Huber loss).
        Supports mixing expected and median depth like MILo.

        Args:
            mesh_depth: (H, W) mesh depth map
            gaussian_depth: (H, W) expected Gaussian depth map
            gaussian_depth_med: (H, W) median Gaussian depth map (optional)
            mask: Optional (H, W) boolean mask for valid pixels
            scale: Scale factor (uses self.depth_scale if None, or scene_scaling)
            depth_ratio: Ratio for median depth in mixing (MILo convention)
                         Final = (1 - depth_ratio) * expected + depth_ratio * median

        Returns:
            loss: Scalar depth consistency loss
        """
        if scale is None:
            scale = self.depth_scale
        if depth_ratio is None:
            depth_ratio = self.depth_ratio

        # Mix expected and median depth (MILo style)
        # MILo: gaussians_depth = (1 - depth_ratio) * expected + depth_ratio * median
        if gaussian_depth_med is not None and depth_ratio > 0.0:
            mixed_depth = (1.0 - depth_ratio) * gaussian_depth + depth_ratio * gaussian_depth_med
        else:
            mixed_depth = gaussian_depth

        # Compute absolute difference
        diff = torch.abs(mesh_depth - mixed_depth)

        # Log loss to suppress outliers
        loss = torch.log(1.0 + diff / scale)

        if mask is not None:
            loss = loss * mask.float()
            loss = loss.sum() / (mask.float().sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def compute_occupancy_loss(
        self,
        model,
        gaussian_depth: torch.Tensor,
        camera,
        mask: torch.Tensor = None,
        max_points: int = 10000
    ) -> torch.Tensor:
        """Compute surface occupancy loss.

        Surface points (from depth map back-projection) should have SDF ≈ 0.
        Loss = mean(|SDF(surface_points)|^2)

        Args:
            model: TetModel with query_sdf method
            gaussian_depth: (H, W) Gaussian depth map
            camera: Camera object
            mask: Optional (H, W) boolean mask for valid pixels
            max_points: Maximum number of points to sample (for efficiency)

        Returns:
            loss: Scalar surface occupancy loss
        """
        # Get 3D surface points from depth map
        surface_points, valid_mask = depth_to_points_world(gaussian_depth, camera)

        if surface_points.shape[0] == 0:
            return torch.tensor(0.0, device=gaussian_depth.device)

        # Apply additional mask if provided
        if mask is not None:
            # Flatten and combine masks
            flat_valid = valid_mask.flatten()
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            flat_mask = (mask > 0.5).flatten()
            combined_flat = flat_valid & flat_mask

            # Re-extract points with combined mask
            H, W = gaussian_depth.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=gaussian_depth.device, dtype=torch.float32),
                torch.arange(W, device=gaussian_depth.device, dtype=torch.float32),
                indexing='ij'
            )

            # Recompute all points and filter by combined mask
            fx = W / (2 * math.tan(camera.fovx / 2.))
            fy = H / (2 * math.tan(camera.fovy / 2.))
            cx, cy = W / 2., H / 2.

            x = (grid_x + 0.5 - cx) / fx
            y = (grid_y + 0.5 - cy) / fy
            z = gaussian_depth
            points_view = torch.stack([x * z, y * z, z], dim=-1)

            view_to_world = torch.inverse(camera.world_view_transform.to(gaussian_depth.device))
            R = view_to_world[:3, :3]
            t = view_to_world[:3, 3]
            points_world = points_view @ R.T + t

            # Filter by combined mask
            combined_mask_2d = combined_flat.reshape(H, W)
            surface_points = points_world[combined_mask_2d]

        if surface_points.shape[0] == 0:
            return torch.tensor(0.0, device=gaussian_depth.device)

        # Subsample if too many points (for efficiency)
        if surface_points.shape[0] > max_points:
            indices = torch.randperm(surface_points.shape[0], device=surface_points.device)[:max_points]
            surface_points = surface_points[indices]

        # Query SDF at surface points
        surface_points = surface_points.detach().requires_grad_(False)
        sdf_values = model.query_sdf(surface_points)  # (N,)

        # Surface occupancy loss: SDF should be ~0 at surface
        loss = (sdf_values ** 2).mean()

        return loss

    def update_occupancy_labels(
        self,
        model,
        mesh: Meshes,
        train_cameras,
        iteration: int,
        force_reset: bool = False,
    ) -> torch.Tensor:
        """Update cached occupancy labels if needed.

        Args:
            model: TetModel with vertices
            mesh: Extracted mesh for depth rendering
            train_cameras: List of all training cameras
            iteration: Current iteration
            force_reset: Force recomputation even if interval not reached

        Returns:
            torch.Tensor: (N,) occupancy labels for internal vertices
        """
        num_verts = model.num_int_verts if hasattr(model, 'num_int_verts') else model.vertices.shape[0]

        # Check if we need to recompute labels
        need_reset = (
            force_reset or
            self._cached_occupancy_labels is None or
            self._occupancy_labels_num_verts != num_verts or
            (iteration - self._occupancy_labels_last_reset_iter) >= self.occupancy_labels_reset_interval
        )

        if need_reset:
            print(f"[MeshRegularizer] Recomputing occupancy labels at iteration {iteration}")
            # Get internal vertices only
            internal_verts = model.vertices[:num_verts]

            # Compute labels
            labels = evaluate_vertex_occupancy(
                points=internal_verts,
                mesh=mesh,
                cameras=train_cameras,
                use_scalable_renderer=False,
                eps=1e-4,
                verbose=True,
            )

            # Cache results
            self._cached_occupancy_labels = labels.detach()
            self._occupancy_labels_last_reset_iter = iteration
            self._occupancy_labels_num_verts = num_verts

            # Log stats
            inside_ratio = (labels < 0.5).float().mean().item()
            print(f"[MeshRegularizer] Occupancy labels: {inside_ratio*100:.1f}% inside, {(1-inside_ratio)*100:.1f}% outside")

        return self._cached_occupancy_labels

    def compute_occupancy_labels_loss(
        self,
        model,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute occupancy labels loss using BCE (MILo-style).

        This loss encourages the SDF to be positive (outside) for points
        labeled as outside. Following MILo, we ONLY compute loss for outside
        points (label > 0.5) because inside labels may be inaccurate due to occlusion.

        Args:
            model: TetModel with query_vert_sdf method
            labels: (N,) occupancy labels (0=inside, 1=outside)

        Returns:
            torch.Tensor: Scalar BCE loss (only for outside points)
        """
        # Query current SDF values for internal vertices
        num_verts = model.num_int_verts if hasattr(model, 'num_int_verts') else model.vertices.shape[0]
        vert_sdf = model.query_vert_sdf()[:num_verts]  # (N,)

        # MILo style: use binary_cross_entropy_with_logits with sdf * scale as logit
        # SDF > 0 (outside) -> sigmoid(sdf*scale) > 0.5 -> should match label=1
        # SDF < 0 (inside) -> sigmoid(sdf*scale) < 0.5 -> should match label=0
        scale = 10.0  # Adjustable, controls sharpness

        # Compute per-element BCE loss (no reduction yet)
        per_element_loss = F.binary_cross_entropy_with_logits(
            vert_sdf * scale,  # logits
            labels,            # targets
            reduction='none'
        )

        # MILo: only compute loss for outside points (label > 0.5)
        # Inside labels may be inaccurate due to occlusion
        outside_mask = (labels > 0.5).float()
        masked_loss = per_element_loss * outside_mask

        # Mean over outside points only
        loss = masked_loss.sum() / (outside_mask.sum() + 1e-8)

        return loss

    def forward(
        self,
        model,
        camera,
        gaussian_depth: torch.Tensor,
        gaussian_depth_med: torch.Tensor = None,  # MILo: median depth for mixing
        gaussian_normal: torch.Tensor = None,     # (3, H, W) 渲染的法线（世界空间）
        normal_source: str = "depth",             # "depth", "depth_med", or "rendered"
        spatial_lr_scale: float = None,  # MILo: spatial_lr_scale for depth loss
        iteration: int = None,
        gt_mask: torch.Tensor = None,  # GT alpha mask，排除背景散乱法线
        train_cameras = None,  # [NEW] All training cameras for occupancy labels
    ) -> dict:
        """Compute mesh regularization losses.

        Args:
            model: TetModel with vertices, indices, query_vert_sdf, query_sdf
            camera: Camera object for rendering
            gaussian_depth: (H, W) expected depth from Gaussian rendering
            gaussian_depth_med: (H, W) median depth from Gaussian rendering (MILo style)
            gaussian_normal: (3, H, W) rendered normal in world space (for normal_source="rendered")
            normal_source: Source for computing gaussian normal:
                - "depth": use expected depth (default)
                - "depth_med": use median depth
                - "rendered": use rendered normal directly (skip depth_to_normal)
            spatial_lr_scale: MILo-style spatial scale for depth loss (camera radius * 1.1)
            iteration: Current training iteration (for scheduling)
            gt_mask: (H, W) or (1, H, W) GT alpha mask，背景为0前景为1
            train_cameras: List of all training cameras (for occupancy labels loss)

        Returns:
            dict with:
                - 'loss': Total weighted loss (0 if not active)
                - 'normal_loss_raw': Unweighted normal loss
                - 'depth_loss_raw': Unweighted depth loss
                - 'occupancy_loss_raw': Unweighted occupancy loss
                - 'occupancy_labels_loss_raw': Unweighted occupancy labels loss
                - 'mesh_normal': (H, W, 3) mesh normal map
                - 'mesh_depth': (H, W) mesh depth map
                - 'gaussian_normal': (3, H, W) Gaussian normal map
                - 'active': Whether regularization was applied
        """
        device = gaussian_depth.device
        result = {
            'loss': torch.tensor(0.0, device=device),
            'normal_loss_raw': torch.tensor(0.0, device=device),
            'depth_loss_raw': torch.tensor(0.0, device=device),
            'occupancy_loss_raw': torch.tensor(0.0, device=device),
            'occupancy_labels_loss_raw': torch.tensor(0.0, device=device),  # [NEW]
            'mesh_normal': None,
            'mesh_depth': None,
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

        # Render mesh normals and depth
        mesh_normal, rasterization_mask, mesh_depth = self.render_mesh_normal(mesh, camera)

        # Compute Gaussian normals based on normal_source
        # mesh_normal 已在相机空间 (view space)，gaussian_normal 也需要在相机空间
        if normal_source == "rendered" and gaussian_normal is not None:
            # 使用渲染的法线，需要从世界空间转到相机空间
            # gaussian_normal: (3, H, W) 世界空间
            # 转换到相机空间: n_view = n_world @ R (R = world_view_transform[:3,:3])
            R = camera.world_view_transform[:3, :3].to(device)
            if gaussian_normal.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                gn_hwc = gaussian_normal.permute(1, 2, 0)
            else:
                gn_hwc = gaussian_normal  # 假设已经是 (H, W, 3)
            gn_view = gn_hwc @ R  # world -> view
            gn_view = F.normalize(gn_view, dim=-1)
            computed_gaussian_normal = gn_view.permute(2, 0, 1)  # (3, H, W)
        elif normal_source == "depth_med" and gaussian_depth_med is not None:
            # 使用中位深度推导法线
            computed_gaussian_normal = self.compute_gaussian_normal(gaussian_depth_med, camera)
        else:
            # 默认：使用期望深度推导法线
            computed_gaussian_normal = self.compute_gaussian_normal(gaussian_depth, camera)

        # Build combined mask
        H, W = gaussian_depth.shape

        # 1. Mesh rasterization mask (mesh渲染到的位置)
        combined_mask = rasterization_mask

        # 2. Gaussian depth valid mask
        gaussian_valid_mask = gaussian_depth > 0
        combined_mask = combined_mask & gaussian_valid_mask

        # 3. GT alpha mask (排除背景散乱法线)
        gt_mask_2d = None
        if gt_mask is not None:
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.squeeze(0)
            gt_mask_2d = gt_mask
            gt_mask_bool = gt_mask > 0.5
            combined_mask = combined_mask & gt_mask_bool

        total_loss = torch.tensor(0.0, device=device)

        # 1. Normal consistency loss
        if self.normal_weight > 0:
            normal_loss = self.compute_normal_consistency_loss(
                mesh_normal, computed_gaussian_normal, combined_mask
            )
            result['normal_loss_raw'] = normal_loss
            total_loss = total_loss + self.normal_weight * normal_loss

        # 2. Depth consistency loss (MILo style: with depth mixing and spatial_lr_scale)
        if self.depth_weight > 0:
            # Use spatial_lr_scale if provided, otherwise use self.depth_scale
            depth_scale = spatial_lr_scale if spatial_lr_scale is not None else self.depth_scale
            depth_loss = self.compute_depth_loss(
                mesh_depth, gaussian_depth, gaussian_depth_med,
                mask=combined_mask, scale=depth_scale
            )
            result['depth_loss_raw'] = depth_loss
            total_loss = total_loss + self.depth_weight * depth_loss

        # 3. Surface occupancy loss
        if self.occupancy_weight > 0:
            occupancy_loss = self.compute_occupancy_loss(
                model, gaussian_depth, camera, gt_mask_2d
            )
            result['occupancy_loss_raw'] = occupancy_loss
            total_loss = total_loss + self.occupancy_weight * occupancy_loss

        # 4. [NEW] Occupancy labels loss (MILo-style)
        if self.occupancy_labels_weight > 0 and train_cameras is not None:
            # Update labels if needed (cached computation)
            labels = self.update_occupancy_labels(
                model, mesh, train_cameras, iteration
            )
            # Compute loss
            occupancy_labels_loss = self.compute_occupancy_labels_loss(model, labels)
            result['occupancy_labels_loss_raw'] = occupancy_labels_loss
            total_loss = total_loss + self.occupancy_labels_weight * occupancy_labels_loss

        result['loss'] = total_loss
        result['mesh_normal'] = mesh_normal
        result['mesh_depth'] = mesh_depth
        result['gaussian_normal'] = computed_gaussian_normal
        result['active'] = True

        return result


def create_mesh_regularizer(config: dict = None) -> MeshRegularizer:
    """Factory function to create mesh regularizer from config.

    Args:
        config: Dictionary with keys:
            - start_iter: int (default 8001)
            - stop_iter: int (default 18000)
            - normal_weight: float (default 0.05)
            - depth_weight: float (default 0.0)
            - occupancy_weight: float (default 0.0)
            - occupancy_labels_weight: float (default 0.0)
            - occupancy_labels_reset_interval: int (default 1000)
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
        depth_weight=config.get('depth_weight', 0.0),
        occupancy_weight=config.get('occupancy_weight', 0.0),
        occupancy_labels_weight=config.get('occupancy_labels_weight', 0.0),
        occupancy_labels_reset_interval=config.get('occupancy_labels_reset_interval', 1000),
        sdf_threshold=config.get('sdf_threshold', 0.0),
        use_opengl=config.get('use_opengl', True),
    )
