"""
MILo-style MT Mesh Processing Utilities

Implements the MT mesh processing pipeline from MILo:
1. filter_large_edges: Filter faces with large edges (distance > scale)
2. collapse_large_edges: Collapse large edge vertices to SDF-smaller endpoint
3. remove_oof_vertices: Remove out-of-field vertices (not visible in any camera)

Reference: MILo/milo/mesh_extract_sdf.py
"""

import torch
import numpy as np
from tqdm import tqdm
import math


def identify_out_of_field_points(points, cameras, verbose=True):
    """
    Identify vertices that are out of field of view in all cameras.

    A vertex is considered "out of field" if it's not visible in ANY camera view.

    Args:
        points: (N, 3) torch.Tensor, vertex positions in world coordinates
        cameras: List of cameras with world_view_transform, fovx, fovy, image_width, image_height
        verbose: Whether to show progress bar

    Returns:
        out_of_field_mask: (N,) bool tensor, True if vertex is out of field in ALL cameras
    """
    device = points.device
    n_points = points.shape[0]

    # Start with all points marked as out of field
    out_of_field_mask = torch.ones(n_points, dtype=torch.bool, device=device)

    iterator = tqdm(cameras, desc="Identifying out of field points") if verbose else cameras

    for camera in iterator:
        H = camera.image_height
        W = camera.image_width

        # Transform points to view space: world_view_transform is W2V.T
        w2v = camera.world_view_transform.T.to(device)  # (4, 4)
        points_h = torch.cat([points, torch.ones(n_points, 1, device=device)], dim=-1)  # (N, 4)
        view_points = (points_h @ w2v.T)[:, :3]  # (N, 3) in view space

        # Project to pixel space
        fx = W / (2 * math.tan(camera.fovx / 2.))
        fy = H / (2 * math.tan(camera.fovy / 2.))

        # Pixel coordinates (with z for depth check)
        z = view_points[:, 2]
        pix_x = (view_points[:, 0] / (z + 1e-8)) * fx + W / 2
        pix_y = (view_points[:, 1] / (z + 1e-8)) * fy + H / 2

        # Check if in view frustum (valid pixel coordinates and positive depth)
        znear = getattr(camera, 'znear', 0.01)
        zfar = getattr(camera, 'zfar', 1000.0)

        valid_mask = (
            (pix_x >= 0) & (pix_x <= W - 1) &
            (pix_y >= 0) & (pix_y <= H - 1) &
            (z > znear) & (z < zfar)
        )

        # If a point is visible in this camera, mark it as in-field
        out_of_field_mask[valid_mask] = False

    return out_of_field_mask


def process_mt_mesh_milo(
    edge_endpoints,
    edge_sdf,
    faces,
    edge_vertex_ids,
    cameras,
    filter_large_edges=True,
    collapse_large_edges=True,
    remove_oof_vertices=True,
    edge_scale_threshold=2.0,
    verbose=True
):
    """
    Process MT mesh using MILo-style pipeline.

    Pipeline:
    1. Interpolate vertex positions from edge endpoints using SDF
    2. (Optional) Filter large edges: remove faces where edge distance > threshold
    3. (Optional) Collapse large edges: move vertex to endpoint with smaller |SDF|
    4. (Optional) Remove out-of-field vertices

    Args:
        edge_endpoints: (N_verts, 2, 3) positions of edge endpoints
        edge_sdf: (N_verts, 2, 1) SDF values at edge endpoints
        faces: (N_faces, 3) face indices
        edge_vertex_ids: (N_verts, 2) vertex indices of edge endpoints
        cameras: List of cameras for OOF check
        filter_large_edges: Whether to filter faces with large edges
        collapse_large_edges: Whether to collapse large edge vertices
        remove_oof_vertices: Whether to remove out-of-field vertices
        edge_scale_threshold: For large edge filtering, threshold = mean_edge_length * this factor
        verbose: Whether to print progress

    Returns:
        verts: (N_verts', 3) processed vertex positions
        faces: (N_faces', 3) processed face indices
        vertex_mask: (N_verts,) bool, which original vertices are kept
        face_mask: (N_faces,) bool, which original faces are kept
    """
    device = edge_endpoints.device
    n_verts = edge_endpoints.shape[0]
    n_faces = faces.shape[0]

    if verbose:
        print(f"[MILo MT] Processing mesh: {n_verts} vertices, {n_faces} faces")

    # Step 1: Interpolate vertex positions using SDF (same as standard MT)
    # v = v0 * |sdf1| / (|sdf0| + |sdf1|) + v1 * |sdf0| / (|sdf0| + |sdf1|)
    sdf_abs = edge_sdf.abs()  # (N, 2, 1)
    norm_sdf = sdf_abs / (sdf_abs.sum(dim=1, keepdim=True) + 1e-10)
    # norm_sdf[:, 0] is weight for v1, norm_sdf[:, 1] is weight for v0
    verts = edge_endpoints[:, 0, :] * norm_sdf[:, 1, :] + edge_endpoints[:, 1, :] * norm_sdf[:, 0, :]  # (N, 3)

    # Initialize masks
    vertex_mask = torch.ones(n_verts, dtype=torch.bool, device=device)
    face_mask = torch.ones(n_faces, dtype=torch.bool, device=device)

    # Step 2 & 3: Edge filtering (filter_large_edges and/or collapse_large_edges)
    if filter_large_edges or collapse_large_edges:
        # Compute edge distances
        edge_distance = torch.norm(edge_endpoints[:, 0, :] - edge_endpoints[:, 1, :], dim=-1)  # (N,)

        # MILo uses Gaussian scales, we use edge length statistics as proxy
        # Threshold = mean_edge_length * edge_scale_threshold
        mean_edge_length = edge_distance.mean()
        edge_threshold = mean_edge_length * edge_scale_threshold

        # Identify large edges
        large_edge_mask = edge_distance > edge_threshold  # (N,) True for large edges

        if verbose:
            n_large = large_edge_mask.sum().item()
            print(f"[MILo MT] Found {n_large}/{n_verts} large edges (threshold={edge_threshold:.6f})")

        if filter_large_edges:
            # Filter faces: only keep faces where ALL vertices have small edges
            # A face is removed if ANY of its vertices is a large edge
            small_edge_vertex_mask = ~large_edge_mask
            face_mask = small_edge_vertex_mask[faces].all(dim=1)

            if verbose:
                print(f"[MILo MT] After filter_large_edges: {face_mask.sum().item()}/{n_faces} faces kept")

        if collapse_large_edges:
            # For large edges, collapse vertex to endpoint with smaller |SDF|
            # Find which endpoint has smaller |SDF|
            min_sdf_idx = sdf_abs.squeeze(-1).argmin(dim=1)  # (N,) 0 or 1

            # Get the endpoint position with smaller SDF
            min_sdf_endpoints = edge_endpoints[
                torch.arange(n_verts, device=device),
                min_sdf_idx
            ]  # (N, 3)

            # Replace vertex position for large edges
            verts = torch.where(
                large_edge_mask.unsqueeze(-1),  # (N, 1)
                min_sdf_endpoints,
                verts
            )

            if verbose:
                print(f"[MILo MT] Collapsed {large_edge_mask.sum().item()} large edge vertices")

    # Step 4: Remove out-of-field vertices
    if remove_oof_vertices and cameras is not None and len(cameras) > 0:
        oof_mask = identify_out_of_field_points(verts, cameras, verbose=verbose)

        # Update vertex mask
        vertex_mask = vertex_mask & (~oof_mask)

        # Update face mask: remove faces with ANY out-of-field vertex
        oof_face_mask = (~oof_mask)[faces].all(dim=1)
        face_mask = face_mask & oof_face_mask

        if verbose:
            print(f"[MILo MT] Removed {oof_mask.sum().item()} out-of-field vertices")
            print(f"[MILo MT] After remove_oof: {face_mask.sum().item()}/{n_faces} faces kept")

    # Apply vertex filtering: need to remap face indices
    if not vertex_mask.all():
        # Create new vertex indices
        new_vertex_indices = torch.zeros(n_verts, dtype=torch.long, device=device)
        new_vertex_indices[vertex_mask] = torch.arange(vertex_mask.sum(), device=device)

        # Filter vertices
        verts = verts[vertex_mask]

        # Remap face indices
        valid_faces = faces[face_mask]
        # Only keep faces where all vertices are kept
        face_vertex_valid = vertex_mask[valid_faces].all(dim=1)
        valid_faces = valid_faces[face_vertex_valid]
        face_mask_final = face_mask.clone()
        face_mask_final[face_mask] = face_vertex_valid
        face_mask = face_mask_final

        faces = new_vertex_indices[valid_faces]
    else:
        faces = faces[face_mask]

    if verbose:
        print(f"[MILo MT] Final mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

    return verts, faces, vertex_mask, face_mask


def extract_mesh_with_milo_processing(
    model,
    cameras,
    sdf_threshold=0.0,
    filter_large_edges=True,
    collapse_large_edges=True,
    remove_oof_vertices=True,
    edge_scale_threshold=2.0,
    verbose=True
):
    """
    Extract mesh using Marching Tetrahedra and apply MILo-style processing.

    Args:
        model: Model with vertices, indices, query_vert_sdf(), num_int_verts
        cameras: List of cameras for OOF check
        sdf_threshold: SDF threshold for surface extraction (default 0.0)
        filter_large_edges: Whether to filter faces with large edges
        collapse_large_edges: Whether to collapse large edge vertices
        remove_oof_vertices: Whether to remove out-of-field vertices
        edge_scale_threshold: For large edge filtering, threshold = mean_edge_length * this factor
        verbose: Whether to print progress

    Returns:
        verts: (N, 3) mesh vertex positions
        faces: (M, 3) mesh face indices
    """
    from utils.tetmesh import marching_tetrahedra_with_interpolation

    with torch.no_grad():
        # Get vertex SDF values
        vert_sdf = model.query_vert_sdf()  # (N_verts,)

        # Apply threshold offset
        vert_sdf = vert_sdf - sdf_threshold

        # Use MT with interpolation to get edge information
        edge_endpoints, edge_sdf, faces, edge_vertex_ids = marching_tetrahedra_with_interpolation(
            model.vertices,
            model.indices,
            vert_sdf,
            num_internal=model.num_int_verts
        )

        if faces.shape[0] == 0:
            if verbose:
                print("[MILo MT] No faces extracted")
            return torch.zeros((0, 3), device=model.device), torch.zeros((0, 3), dtype=torch.long, device=model.device)

        # Apply MILo-style processing
        verts, faces, _, _ = process_mt_mesh_milo(
            edge_endpoints=edge_endpoints,
            edge_sdf=edge_sdf,
            faces=faces,
            edge_vertex_ids=edge_vertex_ids,
            cameras=cameras,
            filter_large_edges=filter_large_edges,
            collapse_large_edges=collapse_large_edges,
            remove_oof_vertices=remove_oof_vertices,
            edge_scale_threshold=edge_scale_threshold,
            verbose=verbose
        )

        return verts, faces
