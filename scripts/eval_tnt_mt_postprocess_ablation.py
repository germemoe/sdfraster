#!/usr/bin/env python3
"""
Evaluate TNT MT meshes with alternative post-processing.

Case 1: No cluster post-processing (raw MT mesh).
Case 2: No cluster + filter_large_edges using per-vertex local edge-length scale.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import open3d as o3d

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.tetmesh import marching_tetrahedra, marching_tetrahedra_with_interpolation
from utils.model_util import extract_edges_from_tets
from eval.tnt.evaluate import evaluate_tnt_mesh, get_tnt_info


def save_mesh(verts_np, faces_np, path: Path) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    o3d.io.write_triangle_mesh(str(path), mesh)


def extract_mt_no_cluster(model):
    with torch.no_grad():
        vert_sdf = model.query_vert_sdf()
        mt_verts, mt_faces = marching_tetrahedra(
            model.vertices, model.indices, vert_sdf, num_internal=model.num_int_verts
        )
    return mt_verts, mt_faces


def extract_mt_edgefilter(model, scale_factor: float = 1.0):
    with torch.no_grad():
        vert_sdf = model.query_vert_sdf()
        edge_endpoints, edge_sdf, faces, edge_vertex_ids = marching_tetrahedra_with_interpolation(
            model.vertices, model.indices, vert_sdf, num_internal=model.num_int_verts
        )

        # Interpolate MT vertices from edge endpoints.
        sdf_abs = edge_sdf.abs()
        norm_sdf = sdf_abs / (sdf_abs.sum(dim=1, keepdim=True) + 1e-10)
        verts = edge_endpoints[:, 0, :] * norm_sdf[:, 1, :] + edge_endpoints[:, 1, :] * norm_sdf[:, 0, :]

        # Local scale per vertex: mean of adjacent tet edge lengths.
        tet_edges = extract_edges_from_tets(model.indices)
        edge_len = (model.vertices[tet_edges[:, 0]] - model.vertices[tet_edges[:, 1]]).norm(dim=1)
        num_verts = model.vertices.shape[0]
        sum_len = torch.zeros(num_verts, device=model.vertices.device)
        count = torch.zeros(num_verts, device=model.vertices.device)
        ones = torch.ones_like(edge_len)
        sum_len.scatter_add_(0, tet_edges[:, 0], edge_len)
        sum_len.scatter_add_(0, tet_edges[:, 1], edge_len)
        count.scatter_add_(0, tet_edges[:, 0], ones)
        count.scatter_add_(0, tet_edges[:, 1], ones)
        local_scale = sum_len / (count + 1e-8)

        edge_len_mt = (edge_endpoints[:, 0, :] - edge_endpoints[:, 1, :]).norm(dim=1)
        edge_scale = (local_scale[edge_vertex_ids[:, 0]] + local_scale[edge_vertex_ids[:, 1]]) * scale_factor
        keep_edge = edge_len_mt <= edge_scale
        face_mask = keep_edge[faces].all(dim=1)
        faces_filtered = faces[face_mask]

    return verts, faces_filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="/home/jk/saids_niu/output/TNT_ablation/normal_depth_med_0.05",
        type=str,
        help="Model output directory containing config.json and ckpt.",
    )
    parser.add_argument(
        "--edge_scale_factor",
        default=1.0,
        type=float,
        help="Scale factor for edge-length filtering threshold.",
    )
    parser.add_argument(
        "--tnt_gt_root",
        default=None,
        type=str,
        help="Optional TNT GT root. Supports flat layout (scene.ply in root) or GT_TNT/scene/scene.ply.",
    )
    parser.add_argument(
        "--mesh_no_cluster",
        default=None,
        type=str,
        help="Optional path to a pre-extracted MT mesh for the no-cluster case.",
    )
    parser.add_argument(
        "--mesh_edgefilter",
        default=None,
        type=str,
        help="Optional path to a pre-extracted MT mesh for the edge-filter case.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = Args.load_from_json(str(model_dir / "config.json"))

    train_cameras, _, _ = loader.load_dataset(
        "/home/jk/saids_niu/dataset/TNT/Barn",
        config.image_folder,
        data_device="cpu",
        eval=False,
        resolution=config.resolution,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    device = torch.device("cuda")
    model = Model.load_ckpt(model_dir, device)
    model.eval()

    tnt_root, scene_name, is_tnt = get_tnt_info("/home/jk/saids_niu/dataset/TNT/Barn")
    if not is_tnt:
        raise RuntimeError("Dataset is not TNT.")

    if args.tnt_gt_root is not None:
        gt_root = Path(args.tnt_gt_root)
        flat_ply = gt_root / f"{scene_name}.ply"
        flat_log = gt_root / f"{scene_name}_COLMAP_SfM.log"
        nested_ply = gt_root / scene_name / f"{scene_name}.ply"
        nested_log = gt_root / scene_name / f"{scene_name}_COLMAP_SfM.log"
        if flat_ply.exists() and flat_log.exists():
            tnt_root = gt_root
        elif nested_ply.exists() and nested_log.exists():
            tnt_root = gt_root / scene_name
        else:
            raise RuntimeError(
                f"Could not find GT files for {scene_name} under {gt_root}."
            )

    min_visible = getattr(config, "tnt_min_visible_views", 20)

    # Case 1: No cluster
    no_cluster_dir = model_dir / "eval_tnt_mt_no_cluster"
    no_cluster_dir.mkdir(exist_ok=True)
    if args.mesh_no_cluster is None:
        print("[Eval] Extracting MT mesh (no cluster)...")
        mt_verts, mt_faces = extract_mt_no_cluster(model)
        mesh_no_cluster = no_cluster_dir / "mesh_mt_no_cluster.ply"
        save_mesh(mt_verts.detach().cpu().numpy(), mt_faces.detach().cpu().numpy(), mesh_no_cluster)
    else:
        mesh_no_cluster = Path(args.mesh_no_cluster)
        print(f"[Eval] Using existing mesh (no cluster): {mesh_no_cluster}")

    print("[Eval] Running TNT eval (no cluster)...")
    res_no_cluster = evaluate_tnt_mesh(
        mesh_path=str(mesh_no_cluster),
        train_cameras=train_cameras,
        tnt_path=str(tnt_root),
        scene_name=scene_name,
        output_dir=str(no_cluster_dir),
        model_output_path=str(model_dir),
        cull=False,
        min_visible_views=min_visible,
    )

    # Case 2: Edge-length scale filter
    edgefilter_dir = model_dir / "eval_tnt_mt_edgefilter"
    edgefilter_dir.mkdir(exist_ok=True)
    if args.mesh_edgefilter is None:
        print("[Eval] Extracting MT mesh (edge-length scale filter)...")
        ef_verts, ef_faces = extract_mt_edgefilter(model, scale_factor=args.edge_scale_factor)
        mesh_edgefilter = edgefilter_dir / "mesh_mt_edgefilter.ply"
        save_mesh(ef_verts.detach().cpu().numpy(), ef_faces.detach().cpu().numpy(), mesh_edgefilter)
    else:
        mesh_edgefilter = Path(args.mesh_edgefilter)
        print(f"[Eval] Using existing mesh (edge filter): {mesh_edgefilter}")

    print("[Eval] Running TNT eval (edge-length scale filter)...")
    res_edgefilter = evaluate_tnt_mesh(
        mesh_path=str(mesh_edgefilter),
        train_cameras=train_cameras,
        tnt_path=str(tnt_root),
        scene_name=scene_name,
        output_dir=str(edgefilter_dir),
        model_output_path=str(model_dir),
        cull=False,
        min_visible_views=min_visible,
    )

    print("\n[Results]")
    print("no_cluster:", res_no_cluster)
    print("edgefilter:", res_edgefilter)


if __name__ == "__main__":
    main()
