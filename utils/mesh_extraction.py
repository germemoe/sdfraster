"""Mesh extraction utilities for the radmesh project.

This module provides functions to extract triangle meshes from the tetrahedral
representation using the Marching Tetrahedra algorithm.
"""

import torch
from utils.tetmesh import marching_tetrahedra, marching_tetrahedra_with_interpolation
from utils.mesh import Meshes, MeshRasterizer, MeshRenderer, save_mesh_as_ply


def extract_mesh_from_model(model, threshold=0.0):
    """Extract triangle mesh from TetModel using Marching Tetrahedra.

    Args:
        model: TetModel or FrozenTetModel with vertices, indices, and query_vert_sdf method
        threshold: SDF threshold for surface extraction (default 0.0)

    Returns:
        Meshes: Extracted triangle mesh
    """
    # Get vertices and tetrahedra indices
    vertices = model.vertices.detach()  # (N_verts, 3)
    tets = model.indices  # (N_tets, 4)

    # Query SDF values at all vertices
    vert_sdf = model.query_vert_sdf().detach()  # (N_verts,)

    # Shift SDF by threshold
    sdf_shifted = vert_sdf - threshold

    # Run marching tetrahedra
    mesh_verts, mesh_faces = marching_tetrahedra(vertices, tets, sdf_shifted)

    if mesh_verts.shape[0] == 0:
        print("[Warning] No surface extracted. Check SDF values and threshold.")
        return None

    return Meshes(verts=mesh_verts, faces=mesh_faces)


def extract_mesh_with_color(model, camera=None, threshold=0.0):
    """Extract triangle mesh with vertex colors from TetModel.

    Args:
        model: TetModel with vertices, indices, query_vert_sdf, and get_cell_values methods
        camera: Camera for color evaluation (optional)
        threshold: SDF threshold for surface extraction

    Returns:
        Meshes: Extracted triangle mesh with vertex colors
    """
    # Get vertices and tetrahedra indices
    vertices = model.vertices.detach()
    tets = model.indices

    # Query SDF values at all vertices
    vert_sdf = model.query_vert_sdf().detach()
    sdf_shifted = vert_sdf - threshold

    # Use detailed extraction to get edge information
    edge_endpoints, edge_sdf, mesh_faces, edge_vertex_ids = marching_tetrahedra_with_interpolation(
        vertices, tets, sdf_shifted
    )

    if edge_endpoints.shape[0] == 0:
        print("[Warning] No surface extracted.")
        return None

    # Interpolate vertex positions
    sdf_abs = edge_sdf.abs()
    weights = sdf_abs / (sdf_abs.sum(dim=1, keepdim=True) + 1e-10)
    mesh_verts = edge_endpoints[:, 0] * weights[:, 1] + edge_endpoints[:, 1] * weights[:, 0]

    # TODO: Add color interpolation if model supports per-vertex colors
    # For now, use a default gray color
    mesh_colors = torch.ones((mesh_verts.shape[0], 3), device=mesh_verts.device) * 0.7

    return Meshes(verts=mesh_verts, faces=mesh_faces, verts_colors=mesh_colors)


def render_mesh(mesh, camera, return_depth=True, return_normals=True, use_opengl=True):
    """Render a mesh from a given camera viewpoint.

    Args:
        mesh: Meshes object to render
        camera: Camera object
        return_depth: Whether to return depth map
        return_normals: Whether to return normal map
        use_opengl: Use OpenGL context (True) or CUDA context (False)

    Returns:
        dict: Rendered outputs containing 'depth' and/or 'normals'
    """
    rasterizer = MeshRasterizer(cameras=[camera], use_opengl=use_opengl)
    renderer = MeshRenderer(rasterizer)

    output = renderer(
        mesh,
        cameras=[camera],
        cam_idx=0,
        return_depth=return_depth,
        return_normals=return_normals,
    )

    return output


def save_extracted_mesh(model, filepath, threshold=0.0):
    """Extract mesh from model and save to PLY file.

    Args:
        model: TetModel with query_vert_sdf method
        filepath: Output PLY file path
        threshold: SDF threshold for surface extraction

    Returns:
        Meshes: The extracted mesh (also saved to file)
    """
    mesh = extract_mesh_from_model(model, threshold=threshold)

    if mesh is not None:
        save_mesh_as_ply(filepath, mesh)
        print(f"Mesh extracted and saved to {filepath}")

    return mesh


# Example usage:
# from utils.mesh_extraction import extract_mesh_from_model, render_mesh, save_extracted_mesh
#
# # Extract mesh from trained model
# mesh = extract_mesh_from_model(model, threshold=0.0)
#
# # Render mesh
# output = render_mesh(mesh, camera)
# depth = output['depth']
# normals = output['normals']
#
# # Save mesh
# save_extracted_mesh(model, 'output/mesh.ply')
