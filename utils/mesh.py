"""Mesh representation and rendering utilities using NVDiffRast.

Adapted from MILo project.
"""

from typing import List, Union, Optional
import torch
import torch.nn as nn

try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    print("[Warning] nvdiffrast not available. Mesh rendering will not work.")


__all__ = [
    'Meshes', 'MeshRasterizer', 'MeshRenderer', 'ScalableMeshRenderer',
    'nvdiff_rasterization', 'combine_meshes', 'save_mesh_as_ply'
]


def transform_points_world_to_view(points, cameras):
    """Transform points from world space to view space.

    Args:
        points: (N, 3) or (n_cameras, N, 3) - point positions
        cameras: Camera or List[Camera]

    Returns:
        torch.Tensor: (n_cameras, N, 3) view-space points
    """
    if not isinstance(cameras, list):
        cameras = [cameras]

    if points.dim() == 2:
        points = points.unsqueeze(0).expand(len(cameras), -1, -1)

    world_view_transforms = torch.stack(
        [camera.world_view_transform for camera in cameras], dim=0
    )  # (n_cameras, 4, 4)

    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    return view_points


def nvdiff_rasterization(
    camera,
    image_height: int,
    image_width: int,
    verts: torch.Tensor,
    faces: torch.Tensor,
    glctx=None,
    return_rast_out: bool = False,
    return_positions: bool = False,
):
    """Rasterize mesh using NVDiffRast.

    Args:
        camera: Camera object with full_proj_transform attribute
        image_height: Output image height
        image_width: Output image width
        verts: (N_verts, 3) vertex positions
        faces: (N_faces, 3) face indices
        glctx: NVDiffRast context (optional)
        return_rast_out: Whether to return raw rasterization output
        return_positions: Whether to return clip-space positions

    Returns:
        tuple: (bary_coords, zbuf, pix_to_face, [rast_out], [pos])
    """
    if not NVDIFFRAST_AVAILABLE:
        raise RuntimeError("nvdiffrast is required for mesh rasterization")

    device = verts.device

    # Get full projection matrix
    camera_mtx = camera.full_proj_transform.to(device)

    # Convert to homogeneous coordinates
    pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)

    # Transform points to NDC/clip space
    pos = torch.matmul(pos, camera_mtx)[None]

    # Rasterize with NVDiffRast
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])
    bary_coords, zbuf, pix_to_face = rast_out[..., :2], rast_out[..., 2], rast_out[..., 3].int() - 1

    _output = (bary_coords, zbuf, pix_to_face)
    if return_rast_out:
        _output = _output + (rast_out,)
    if return_positions:
        _output = _output + (pos,)
    return _output


class Meshes(nn.Module):
    """Mesh representation class storing vertices, faces, and optional colors."""

    def __init__(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        verts_colors: torch.Tensor = None
    ):
        super().__init__()
        assert verts_colors is None or verts_colors.shape[0] == verts.shape[0]
        self.verts = verts
        self.faces = faces.to(torch.int32)
        self.verts_colors = verts_colors

    @property
    def face_normals(self):
        """Compute face normals."""
        faces_verts = self.verts[self.faces]
        faces_verts_normals = torch.cross(
            faces_verts[:, 1] - faces_verts[:, 0],
            faces_verts[:, 2] - faces_verts[:, 0],
            dim=-1
        )
        faces_verts_normals = torch.nn.functional.normalize(faces_verts_normals, dim=-1)
        return faces_verts_normals

    @property
    def vertex_normals(self):
        """Compute per-vertex normals by averaging incident face normals."""
        face_normals = self.face_normals  # (N_faces, 3)
        vertex_normals = torch.zeros_like(self.verts)

        # Accumulate face normals to vertices
        vertex_normals.index_add_(0, self.faces[:, 0].long(), face_normals)
        vertex_normals.index_add_(0, self.faces[:, 1].long(), face_normals)
        vertex_normals.index_add_(0, self.faces[:, 2].long(), face_normals)

        # Normalize
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)
        return vertex_normals

    def submesh(
        self,
        vert_idx: Optional[torch.Tensor] = None,
        face_idx: Optional[torch.Tensor] = None,
        vert_mask: Optional[torch.Tensor] = None,
        face_mask: Optional[torch.Tensor] = None,
    ):
        """Extract a submesh by vertex or face selection."""
        assert (
            (vert_idx is not None) or (vert_mask is not None)
            or
            (face_idx is not None) or (face_mask is not None)
        ), "Either vert_idx, vert_mask, face_idx, or face_mask must be provided"

        if (vert_idx is not None) or (vert_mask is not None):
            if vert_mask is None:
                vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
                vert_mask[vert_idx] = True
            face_mask = vert_mask[self.faces].all(dim=1)

        elif (face_idx is not None) or (face_mask is not None):
            if face_mask is None:
                face_mask = torch.zeros(self.faces.shape[0], dtype=torch.bool, device=self.verts.device)
                face_mask[face_idx] = True
            vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
            vert_mask[self.faces[face_mask]] = True

        old_vert_idx_to_new_vert_idx = torch.zeros(
            self.verts.shape[0], dtype=self.faces.dtype, device=self.verts.device
        )
        old_vert_idx_to_new_vert_idx[vert_mask] = torch.arange(
            vert_mask.sum(), dtype=self.faces.dtype, device=self.verts.device
        )

        new_verts = self.verts[vert_mask]
        new_verts_colors = None if self.verts_colors is None else self.verts_colors[vert_mask]
        new_faces = old_vert_idx_to_new_vert_idx[self.faces][face_mask]

        return Meshes(verts=new_verts, faces=new_faces, verts_colors=new_verts_colors)


def combine_meshes(meshes: List[Meshes]) -> Meshes:
    """Combine multiple meshes into a single mesh."""
    all_verts = torch.zeros(0, 3, dtype=meshes[0].verts.dtype, device=meshes[0].verts.device)
    all_faces = torch.zeros(0, 3, dtype=meshes[0].faces.dtype, device=meshes[0].faces.device)
    all_verts_colors = (
        None if meshes[0].verts_colors is None
        else torch.zeros(0, 3, dtype=meshes[0].verts_colors.dtype, device=meshes[0].verts_colors.device)
    )

    n_total_verts = 0

    for mesh in meshes:
        all_verts = torch.cat([all_verts, mesh.verts], dim=0)
        all_faces = torch.cat([all_faces, mesh.faces + n_total_verts], dim=0)
        if all_verts_colors is not None:
            all_verts_colors = torch.cat([all_verts_colors, mesh.verts_colors], dim=0)
        n_total_verts += mesh.verts.shape[0]

    return Meshes(verts=all_verts, faces=all_faces, verts_colors=all_verts_colors)


class Fragments:
    """Rasterization fragments containing barycentric coords, depth, and face indices."""

    def __init__(self, bary_coords, zbuf, pix_to_face):
        self.bary_coords = bary_coords  # Shape (1, height, width, 1, 3)
        self.zbuf = zbuf  # Shape (1, height, width, 1)
        self.pix_to_face = pix_to_face  # Shape (1, height, width, 1)


class RasterizationSettings:
    """Rasterization settings."""

    def __init__(
        self,
        image_size=(1080, 1920),
        blur_radius=0.0,
        faces_per_pixel=1,
    ):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel


class MeshRasterizer(nn.Module):
    """Mesh rasterizer using NVDiffRast."""

    def __init__(
        self,
        cameras=None,
        raster_settings: RasterizationSettings = None,
        use_opengl=True,
    ):
        super().__init__()

        if not NVDIFFRAST_AVAILABLE:
            raise RuntimeError("nvdiffrast is required for MeshRasterizer")

        if cameras is None:
            if raster_settings is None:
                raster_settings = RasterizationSettings()
            self.raster_settings = raster_settings
            self.height, self.width = raster_settings.image_size
            self.cameras = None
        else:
            if not isinstance(cameras, list):
                cameras = [cameras]
            self.height = cameras[0].image_height
            self.width = cameras[0].image_width
            self.raster_settings = RasterizationSettings(
                image_size=(self.height, self.width),
            )
            self.cameras = cameras

        if use_opengl:
            self.gl_context = dr.RasterizeGLContext()
        else:
            self.gl_context = dr.RasterizeCudaContext()

    def forward(
        self,
        mesh: Meshes,
        cameras=None,
        cam_idx: int = 0,
        return_only_pix_to_face: bool = False,
        return_rast_out: bool = False,
        return_positions: bool = False,
    ):
        if cameras is None:
            if self.cameras is None:
                raise ValueError("cameras must be provided either in the constructor or in the forward method")
            cameras = self.cameras

        if not isinstance(cameras, list):
            render_camera = cameras
        else:
            render_camera = cameras[cam_idx]

        height, width = render_camera.image_height, render_camera.image_width
        nvdiff_rast_out = nvdiff_rasterization(
            camera=render_camera,
            image_height=height,
            image_width=width,
            verts=mesh.verts,
            faces=mesh.faces,
            glctx=self.gl_context,
            return_rast_out=return_rast_out,
            return_positions=return_positions,
        )
        bary_coords, zbuf, pix_to_face = nvdiff_rast_out[:3]
        rast_out = nvdiff_rast_out[3] if return_rast_out else None
        pos = nvdiff_rast_out[4] if return_positions else None

        if return_only_pix_to_face:
            return pix_to_face.view(1, height, width, 1)

        bary_coords = torch.cat([bary_coords, 1. - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)

        fragments = Fragments(
            bary_coords.view(1, height, width, 1, 3),
            zbuf.view(1, height, width, 1),
            pix_to_face.view(1, height, width, 1),
        )

        _output = (fragments,)
        if return_rast_out:
            _output = _output + (rast_out,)
        if return_positions:
            _output = _output + (pos,)
        return _output


class MeshRenderer(nn.Module):
    """Mesh renderer using NVDiffRast with interpolation and antialiasing."""

    def __init__(self, rasterizer: MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(
        self,
        mesh: Meshes,
        cameras=None,
        cam_idx=0,
        return_depth=False,
        return_normals=False,
        use_antialiasing=True,
        return_pix_to_face=False,
    ):
        fragments, rast_out, pos = self.rasterizer(
            mesh, cameras, cam_idx, return_rast_out=True, return_positions=True
        )

        if cameras is None:
            cameras = self.rasterizer.cameras
        if not isinstance(cameras, list):
            cameras = [cameras]

        return_colors = mesh.verts_colors is not None
        output_pkg = {}

        # Compute per-vertex features to render
        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)

        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(
                mesh.verts, [cameras[cam_idx]]
            )[..., 2].squeeze()  # Shape (N,)
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)

        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)

        # Compute image via interpolation (only if we have features to interpolate)
        if features.shape[-1] > 0:
            feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)

            # Antialiasing for propagating gradients
            if use_antialiasing:
                feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)

            if return_depth:
                output_pkg["depth"] = feature_img[..., depth_idx:depth_idx + 1]

            if return_colors:
                output_pkg["rgb"] = feature_img[..., color_idx:color_idx + 3]

        # Compute per-face normals
        if return_normals:
            valid_mask = fragments.pix_to_face >= 0
            error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
            if torch.sum(error_mask) > 0:
                print(f"[WARNING] Rasterized {torch.sum(error_mask)} pixels with invalid triangle index.")
                fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask

        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face

        return output_pkg


def fuse_fragments(fragments1: Fragments, fragments2: Fragments):
    """Fuse two fragment buffers, keeping the closer fragment at each pixel."""
    raster_mask1 = fragments1.pix_to_face > -1
    raster_mask2 = fragments2.pix_to_face > -1

    no_raster_mask = (~raster_mask1) & (~raster_mask2)

    zbuf1 = torch.where(raster_mask1, fragments1.zbuf, 1000.)
    zbuf2 = torch.where(raster_mask2, fragments2.zbuf, 1000.)

    all_zbufs = torch.cat([zbuf1[..., None], zbuf2[..., None]], dim=-1)
    zbuf, argzbuf = torch.min(all_zbufs, dim=-1)
    zbuf[no_raster_mask] = 0.

    all_pix_to_face = torch.cat(
        [fragments1.pix_to_face[..., None], fragments2.pix_to_face[..., None]],
        dim=-1
    )
    pix_to_face = torch.gather(all_pix_to_face, dim=-1, index=argzbuf[..., None])[..., 0]
    pix_to_face[no_raster_mask] = -1

    all_bary_coords = torch.cat(
        [fragments1.bary_coords[..., None], fragments2.bary_coords[..., None]], dim=-1
    )
    bary_coords = torch.gather(
        all_bary_coords, dim=-1,
        index=argzbuf[..., None, None].expand(-1, -1, -1, -1, 3, -1)
    )[..., 0]

    return Fragments(bary_coords, zbuf, pix_to_face)


class ScalableMeshRenderer(nn.Module):
    """Scalable mesh renderer for large meshes using chunked rasterization."""

    def __init__(self, rasterizer: MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(
        self,
        mesh: Meshes,
        cameras=None,
        cam_idx: int = 0,
        return_depth: bool = False,
        return_normals: bool = False,
        use_antialiasing: bool = True,
        return_pix_to_face: bool = False,
        max_triangles_in_batch: int = 2 ** 24
    ):
        n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch

        fragments = None
        idx_shift = 0
        pos = None

        for i_pass in range(n_passes):
            start_idx = i_pass * max_triangles_in_batch
            end_idx = min(start_idx + max_triangles_in_batch, mesh.faces.shape[0])

            sub_faces = mesh.faces[start_idx:end_idx]
            sub_mesh = Meshes(verts=mesh.verts, faces=sub_faces)

            _fragments, _, pos = self.rasterizer(
                sub_mesh, cameras, cam_idx, return_rast_out=True, return_positions=True
            )

            if fragments is None:
                fragments = _fragments
            else:
                _fragments.pix_to_face = torch.where(
                    _fragments.pix_to_face > -1,
                    _fragments.pix_to_face + idx_shift,
                    _fragments.pix_to_face
                )
                fragments = fuse_fragments(fragments, _fragments)

            idx_shift = idx_shift + len(sub_faces)

        # Filter mesh to keep only rasterized faces
        filtered_face_idx, filtered_pix_to_face = fragments.pix_to_face.unique(return_inverse=True)
        filtered_face_idx = filtered_face_idx[1:]
        filtered_faces = mesh.faces[filtered_face_idx]
        filtered_pix_to_face = filtered_pix_to_face - 1
        mesh = Meshes(verts=mesh.verts, faces=filtered_faces, verts_colors=mesh.verts_colors)
        fragments.pix_to_face = filtered_pix_to_face

        # Rebuild rast_out
        rast_out = torch.zeros(*fragments.zbuf.shape[:-1], 4, device=fragments.zbuf.device)
        rast_out[..., :2] = fragments.bary_coords[..., 0, :2]
        rast_out[..., 2:3] = fragments.zbuf
        rast_out[..., 3:4] = fragments.pix_to_face.float() + 1

        if cameras is None:
            cameras = self.rasterizer.cameras
        if not isinstance(cameras, list):
            cameras = [cameras]

        return_colors = mesh.verts_colors is not None
        output_pkg = {}

        features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)

        if return_depth:
            depth_idx = features.shape[-1]
            verts_depth = transform_points_world_to_view(
                mesh.verts, [cameras[cam_idx]]
            )[..., 2].squeeze()
            features = torch.cat([features, verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)

        if return_colors:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh.verts_colors], dim=-1)

        feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)

        if use_antialiasing and pos is not None:
            feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)

        if return_depth:
            output_pkg["depth"] = feature_img[..., depth_idx:depth_idx + 1]

        if return_colors:
            output_pkg["rgb"] = feature_img[..., color_idx:color_idx + 3]

        if return_normals:
            valid_mask = fragments.pix_to_face >= 0
            error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
            if torch.sum(error_mask) > 0:
                print(f"[WARNING] Rasterized {torch.sum(error_mask)} pixels with invalid triangle index.")
                fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                valid_mask = valid_mask & ~error_mask
            output_pkg["normals"] = mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask

        if return_pix_to_face:
            output_pkg["pix_to_face"] = fragments.pix_to_face

        return output_pkg


def save_mesh_as_ply(filepath: str, mesh: Meshes):
    """Save mesh to PLY file format.

    Args:
        filepath: Output file path
        mesh: Meshes object to save
    """
    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    has_colors = mesh.verts_colors is not None

    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices
        if has_colors:
            colors = (mesh.verts_colors.detach().cpu().numpy() * 255).astype(int).clip(0, 255)
            for i, v in enumerate(verts):
                f.write(f"{v[0]} {v[1]} {v[2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
        else:
            for v in verts:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

        # Faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Mesh saved to {filepath} ({len(verts)} vertices, {len(faces)} faces)")
