### adapted from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
### and from MILo project

# Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

__all__ = ['marching_tetrahedra', 'marching_tetrahedra_with_interpolation']

# Lookup table for triangle configurations
# Each row corresponds to one of the 16 possible vertex occupancy patterns (2^4)
# Values are edge indices, -1 means no triangle
triangle_table = torch.tensor([
    [-1, -1, -1, -1, -1, -1],
    [1, 0, 2, -1, -1, -1],
    [4, 0, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=torch.long)

num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)

# Tetrahedron edges: pairs of vertex indices (0-3)
# Edge 0: (0,1), Edge 1: (0,2), Edge 2: (0,3), Edge 3: (1,2), Edge 4: (1,3), Edge 5: (2,3)
base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))


def _unbatched_marching_tetrahedra(vertices, tets, sdf):
    """Unbatched marching tetrahedra - extracts triangle mesh from SDF on tetrahedra.

    Args:
        vertices: (N_verts, 3) - vertex positions
        tets: (N_tets, 4) - tetrahedron vertex indices
        sdf: (N_verts,) - SDF values at each vertex (positive = outside, negative = inside)

    Returns:
        verts: (N_mesh_verts, 3) - mesh vertex positions
        faces: (N_faces, 3) - mesh face indices
    """
    device = vertices.device

    # Handle large meshes by chunking
    chunk_size = 32 * 1024 * 1024
    if tets.shape[0] > chunk_size:
        merged_verts = None
        merged_faces = None
        merged_verts_ids = None

        for tet_chunk in torch.chunk(tets, tets.shape[0] // chunk_size + 1):
            torch.cuda.empty_cache()
            verts, faces, verts_ids = _unbatched_marching_tetrahedra(vertices, tet_chunk, sdf)

            if merged_verts is None:
                merged_verts = verts
                merged_faces = faces
                merged_verts_ids = verts_ids
            else:
                # Merge by finding unique edges
                all_edges = torch.cat([merged_verts_ids, verts_ids], dim=0)
                unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

                # Merge vertices
                unique_verts = torch.zeros((unique_edges.shape[0], 3), dtype=torch.float, device=device)
                unique_verts[idx_map[:merged_verts.shape[0]]] = merged_verts
                unique_verts[idx_map[merged_verts.shape[0]:]] = verts

                # Merge faces
                unique_faces_0 = idx_map[merged_faces.reshape(-1)].reshape(-1, 3)
                unique_faces_1 = idx_map[faces.reshape(-1) + merged_verts.shape[0]].reshape(-1, 3)

                merged_faces = torch.cat([unique_faces_0, unique_faces_1], dim=0)
                merged_verts = unique_verts
                merged_verts_ids = unique_edges
                torch.cuda.empty_cache()

        return merged_verts, merged_faces, merged_verts_ids

    with torch.no_grad():
        # Determine which vertices are inside (sdf < 0) or outside (sdf > 0)
        # Here we use sdf > 0 as "occupied" to match MILo convention
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)

        # Valid tetrahedra are those with some vertices inside and some outside
        valid_tets = (occ_sum > 0) & (occ_sum < 4)

        if valid_tets.sum() == 0:
            # No surface intersection
            return (
                torch.zeros((0, 3), device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 2), dtype=torch.long, device=device)
            )

        # Get all edges from valid tetrahedra
        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)

        # Sort edges so smaller index comes first
        order = (all_edges[:, 0] > all_edges[:, 1]).bool()
        all_edges[order] = all_edges[order][:, [1, 0]]

        # Find unique edges
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        unique_edges = unique_edges.long()

        # Only keep edges that cross the surface (one endpoint inside, one outside)
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        # Create mapping from unique edges to crossing edges
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges]

    # Interpolate vertex positions based on SDF values
    edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)

    # Linear interpolation: find zero-crossing point
    # v = v0 + t * (v1 - v0) where t = |sdf0| / (|sdf0| + |sdf1|)
    sdf_abs = edges_to_interp_sdf.abs()
    weights = sdf_abs / (sdf_abs.sum(dim=1, keepdim=True) + 1e-10)
    # weights[:, 0] is weight for v1, weights[:, 1] is weight for v0
    verts = edges_to_interp[:, 0] * weights[:, 1] + edges_to_interp[:, 1] * weights[:, 0]

    idx_map = idx_map.reshape(-1, 6)

    # Use lookup table to generate triangles
    tetindex = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[tetindex]
    triangle_table_device = triangle_table.to(device)

    # Generate triangle indices
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
    ), dim=0)

    return verts, faces, interp_v


def marching_tetrahedra(vertices, tets, sdf):
    """Convert SDF on tetrahedral mesh to triangle mesh using marching tetrahedra.

    This is a differentiable mesh extraction algorithm.

    Args:
        vertices: (N_verts, 3) or (batch, N_verts, 3) - vertex positions
        tets: (N_tets, 4) - tetrahedron vertex indices
        sdf: (N_verts,) or (batch, N_verts) - SDF values (positive = outside, negative = inside)

    Returns:
        verts: (N_mesh_verts, 3) - interpolated mesh vertex positions
        faces: (N_faces, 3) - mesh face indices

    Note:
        The surface is extracted at sdf=0. Vertices with sdf>0 are considered outside,
        vertices with sdf<0 are considered inside the surface.
    """
    # Handle batched input
    if vertices.dim() == 3:
        results = [_unbatched_marching_tetrahedra(vertices[b], tets, sdf[b])
                   for b in range(vertices.shape[0])]
        return list(zip(*results))[:2]  # Return only verts and faces
    else:
        verts, faces, _ = _unbatched_marching_tetrahedra(vertices, tets, sdf)
        return verts, faces


def marching_tetrahedra_with_interpolation(vertices, tets, sdf):
    """Same as marching_tetrahedra but returns edge information for custom interpolation.

    This is useful when you want to compute vertex positions differently
    (e.g., using learned interpolation weights).

    Args:
        vertices: (N_verts, 3) - vertex positions
        tets: (N_tets, 4) - tetrahedron vertex indices
        sdf: (N_verts,) - SDF values

    Returns:
        edge_endpoints: (N_mesh_verts, 2, 3) - positions of edge endpoints
        edge_sdf: (N_mesh_verts, 2, 1) - SDF values at edge endpoints
        faces: (N_faces, 3) - mesh face indices
        edge_vertex_ids: (N_mesh_verts, 2) - vertex indices of edge endpoints
    """
    device = vertices.device

    with torch.no_grad():
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)

        valid_tets = (occ_sum > 0) & (occ_sum < 4)

        if valid_tets.sum() == 0:
            return (
                torch.zeros((0, 2, 3), device=device),
                torch.zeros((0, 2, 1), device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 2), dtype=torch.long, device=device)
            )

        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)

        order = (all_edges[:, 0] > all_edges[:, 1]).bool()
        all_edges[order] = all_edges[order][:, [1, 0]]

        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        unique_edges = unique_edges.long()

        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        interp_v = unique_edges[mask_edges]

    edge_endpoints = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edge_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)

    idx_map = idx_map.reshape(-1, 6)

    tetindex = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[tetindex]
    triangle_table_device = triangle_table.to(device)

    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
    ), dim=0)

    return edge_endpoints, edge_sdf, faces, interp_v
