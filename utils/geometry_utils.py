"""Geometry utility functions for mesh regularization.

Includes functions for computing normals from depth maps and
depth-to-normal conversion for mesh supervision.
"""

import math
import torch
import torch.nn.functional as F


def depths_to_points(camera, depth_map):
    """Convert depth map to 3D points in view space.

    Args:
        camera: Camera object with fovx, fovy, image_width, image_height
        depth_map: (H, W) depth values

    Returns:
        points: (3, H, W) 3D points in view space
    """
    H, W = camera.image_height, camera.image_width
    device = depth_map.device

    # Compute focal lengths from FOV
    fx = W / (2 * math.tan(camera.fovx / 2.))
    fy = H / (2 * math.tan(camera.fovy / 2.))

    # Inverse intrinsics
    intrins_inv = torch.tensor([
        [1/fx, 0., -W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy)],
        [0., 0., 1.0]
    ], dtype=torch.float32, device=device)

    # Create pixel grid (centered at pixel centers)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device) + 0.5,
        torch.arange(H, device=device) + 0.5,
        indexing='xy'
    )

    # Homogeneous pixel coordinates
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1)

    # Ray directions in view space
    rays_d = intrins_inv @ points

    # Scale by depth
    points_3d = depth_map.reshape(1, -1) * rays_d

    return points_3d.reshape(3, H, W)


def points_to_normal(points):
    """Compute normal map from 3D points using finite differences.

    Args:
        points: (3, H, W) 3D points

    Returns:
        normal_map: (3, H, W) normalized surface normals
    """
    # Finite differences (MILo convention)
    # dx: along H direction (rows)
    # dy: along W direction (columns)
    dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]    # H direction
    dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]    # W direction

    # Cross product: cross(H, W) = cross(dx, dy) - MILo style
    normal = torch.cross(dx, dy, dim=0)

    # Normalize
    normal = F.normalize(normal, dim=0)

    # Pad to original size
    output = torch.zeros_like(points)
    output[:, 1:-1, 1:-1] = normal

    return output


def depth_to_normal(camera, depth_map):
    """Convert depth map to normal map.

    Args:
        camera: Camera object
        depth_map: (H, W) or (1, H, W) or (1, 1, H, W) depth values

    Returns:
        normal_map: (3, H, W) surface normals in view space
    """
    # Handle different input shapes
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(0).squeeze(0)
    elif depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)

    points = depths_to_points(camera, depth_map)
    normal = points_to_normal(points)

    return normal


def depth_double_to_normal(camera, depth1, depth2=None):
    """Convert one or two depth maps to normal maps.

    Args:
        camera: Camera object
        depth1: First depth map (H, W) or (1, H, W)
        depth2: Optional second depth map

    Returns:
        If depth2 is None: (3, H, W) normal map
        If depth2 is provided: (2, 3, H, W) two normal maps stacked
    """
    normal1 = depth_to_normal(camera, depth1)

    if depth2 is None:
        return normal1
    else:
        normal2 = depth_to_normal(camera, depth2)
        return torch.stack([normal1, normal2], dim=0)


def fix_normal_map(camera, normal, normal_in_view_space=True):
    """Fix normal map to ensure normals point towards the camera (MILo style).

    This function flips normals that are pointing away from the camera.
    Important for mesh normals where face winding may be inconsistent.

    Args:
        camera: Camera object with fovx, fovy, image_width, image_height
        normal: (3, H, W) or (H, W, 3) normal map
        normal_in_view_space: If True, normals are already in view space.
                              If False, normals are in world space and will be transformed.

    Returns:
        fixed_normal: Normal map with same shape as input, normals pointing towards camera
    """
    H, W = camera.image_height, camera.image_width
    device = normal.device

    # Compute focal lengths from FOV
    fx = W / (2 * math.tan(camera.fovx / 2.))
    fy = H / (2 * math.tan(camera.fovy / 2.))

    # Inverse intrinsics
    intrins_inv = torch.tensor([
        [1/fx, 0., -W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy)],
        [0., 0., 1.0]
    ], dtype=torch.float32, device=device)

    # Create pixel grid (centered at pixel centers)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device) + 0.5,
        torch.arange(H, device=device) + 0.5,
        indexing='xy'
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float()

    # Ray directions in view space (pointing from camera into scene)
    rays_d = (intrins_inv @ points).reshape(3, H, W)

    # Handle input shape
    input_shape_is_chw = (normal.shape[0] == 3)
    if not input_shape_is_chw:
        # (H, W, 3) -> (3, H, W)
        normal = normal.permute(2, 0, 1)

    # Transform to view space if needed
    if normal_in_view_space:
        normal_view = normal
    else:
        # Transform from world to view space
        normal_view = normal.clone()
        # (3, H, W) -> (H, W, 3)
        normal_view = normal_view.permute(1, 2, 0)
        R = camera.world_view_transform[:3, :3].to(device)
        normal_view = normal_view @ R
        # (H, W, 3) -> (3, H, W)
        normal_view = normal_view.permute(2, 0, 1)

    # Compute sign: normals should point towards camera (opposite to ray direction)
    # If dot(normal, rays_d) > 0, normal points away from camera, need to flip
    # We want dot(normal, -rays_d) > 0, i.e., dot(-rays_d, normal) > 0
    dot_product = (-rays_d * normal_view).sum(dim=0, keepdim=True)  # (1, H, W)
    sign = torch.sign(dot_product)  # (1, H, W)

    # Flip normals where needed
    fixed_normal = sign * normal_view  # (3, H, W)

    # Restore original shape if needed
    if not input_shape_is_chw:
        fixed_normal = fixed_normal.permute(1, 2, 0)  # (H, W, 3)

    return fixed_normal


def transform_normal_world_to_view(normal_world, camera):
    """Transform normals from world space to view space.

    Args:
        normal_world: (..., 3) normals in world space
        camera: Camera object with world_view_transform

    Returns:
        normal_view: (..., 3) normals in view space
    """
    device = normal_world.device

    # Get rotation part of world_view_transform (3x3)
    # world_view_transform[:3,:3] stores R (cam-to-world rotation)
    # To transform world->view, we need R^T: n_view = n_world @ R^T
    R = camera.world_view_transform[:3, :3].to(device)

    # Transform normals: world -> view
    original_shape = normal_world.shape
    normal_flat = normal_world.reshape(-1, 3)
    normal_view = normal_flat @ R.T  # [FIX] Use R.T instead of R

    return normal_view.reshape(original_shape)


def compute_normal_consistency_loss(normal1, normal2, mask=None):
    """Compute normal consistency loss between two normal maps (MILo style).

    Uses cosine similarity: loss = 1 - dot(n1, n2)
    MILo does NOT use absolute value - expects normals to be in same direction.

    Args:
        normal1: (3, H, W) or (H, W, 3) first normal map
        normal2: (3, H, W) or (H, W, 3) second normal map
        mask: Optional (H, W) boolean mask for valid pixels

    Returns:
        loss: Scalar normal consistency loss
    """
    # Ensure shape is (H, W, 3)
    if normal1.shape[0] == 3:
        normal1 = normal1.permute(1, 2, 0)
    if normal2.shape[0] == 3:
        normal2 = normal2.permute(1, 2, 0)

    # Compute dot product
    dot_product = (normal1 * normal2).sum(dim=-1)  # (H, W)

    # Loss: 1 - cos(angle) (MILo style, no abs)
    loss = 1.0 - dot_product

    if mask is not None:
        loss = loss * mask.float()
        loss = loss.sum() / (mask.float().sum() + 1e-8)
    else:
        loss = loss.mean()

    return loss
