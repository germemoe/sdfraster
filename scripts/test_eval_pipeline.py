#!/usr/bin/env python3
"""
Test script to verify the new evaluation pipeline works correctly.

Tests both MT and TSDF paths with an existing trained model.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import open3d as o3d

# Model and dataset paths
MODEL_PATH = "/home/jk/data/project/radmesh/newoutput/sdf/DTU/test/just_tv1e-3_normwdmed_meshdmed_hash22_scan65_fixext2/ckpt.pth"
DATASET_PATH = Path("/storage/jk/dataset/DTU/scan65")
DTU_ROOT = "/storage/jk/dataset/DTU"
SCAN_ID = 65
OUTPUT_DIR = Path("/data/jk/project/radmesh/test_eval_output")

def main():
    print("=" * 60)
    print("Testing New Evaluation Pipeline")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print(f"\n[1] Loading dataset from {DATASET_PATH}...")
    from data import loader
    train_cameras, test_cameras, scene_info = loader.load_dataset(
        DATASET_PATH,
        'images',  # image folder
        data_device='cpu',
        eval=True,
        resolution=1.0
    )
    print(f"    Loaded {len(train_cameras)} training cameras, {len(test_cameras)} test cameras")

    # Load model
    print(f"\n[2] Loading model from {MODEL_PATH}...")
    from models.ingp_color import Model

    device = torch.device('cuda')
    model = Model.load_ckpt(Path(MODEL_PATH).parent, device)
    print(f"    Model loaded: {len(model)} vertices, {model.indices.shape[0]} tetrahedra")

    # ========================================
    # Path 1: MT Extraction + Evaluation
    # ========================================
    print("\n" + "=" * 60)
    print("[MT] Testing Marching Tetrahedra path...")
    print("=" * 60)

    from utils.tetmesh import marching_tetrahedra

    with torch.no_grad():
        # Query vertex SDF
        vert_sdf = model.query_vert_sdf()
        print(f"    Vertex SDF: min={vert_sdf.min():.4f}, max={vert_sdf.max():.4f}")

        # Extract mesh
        mt_verts, mt_faces = marching_tetrahedra(
            model.vertices, model.indices, vert_sdf,
            num_internal=model.num_int_verts
        )

    if mt_faces.shape[0] > 0:
        verts_np = mt_verts.cpu().numpy()
        faces_np = mt_faces.cpu().numpy()
        print(f"    Raw MT mesh: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # Post-process
        def post_process_mesh(verts, faces, num_cluster=50):
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
                triangle_clusters, cluster_n_triangles, _ = mesh_o3d.cluster_connected_triangles()

            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)

            if len(cluster_n_triangles) == 0:
                return verts, faces

            threshold = max(np.sort(cluster_n_triangles)[-num_cluster] if len(cluster_n_triangles) >= num_cluster else 0, 50)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
            mesh_o3d.remove_triangles_by_mask(triangles_to_remove)
            mesh_o3d.remove_unreferenced_vertices()
            mesh_o3d.remove_degenerate_triangles()

            return np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles)

        verts_np, faces_np = post_process_mesh(verts_np, faces_np, num_cluster=50)
        print(f"    After post-process: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # Save mesh
        mt_mesh_path = OUTPUT_DIR / "mesh_mt.ply"
        mt_mesh = o3d.geometry.TriangleMesh()
        mt_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        mt_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        o3d.io.write_triangle_mesh(str(mt_mesh_path), mt_mesh)
        print(f"    Saved: {mt_mesh_path}")

        # Run DTU evaluation
        print(f"\n    Running DTU evaluation...")
        from eval.dtu.evaluate import evaluate_dtu_mesh

        mt_results = evaluate_dtu_mesh(
            mesh_path=str(mt_mesh_path),
            train_cameras=train_cameras,
            dtu_path=DTU_ROOT,
            scan_id=SCAN_ID,
            output_dir=str(OUTPUT_DIR / 'eval_mt'),
            cull=True,
            dilation_radius=6,
            dataset_path=str(DATASET_PATH),  # 传入 dataset_path 以使用 scale_mat 对齐
        )

        if mt_results:
            print(f"\n[MT] Results:")
            print(f"    mean_d2s: {mt_results['mean_d2s']:.4f}")
            print(f"    mean_s2d: {mt_results['mean_s2d']:.4f}")
            print(f"    overall:  {mt_results['overall']:.4f}")
        else:
            print("[MT] Evaluation failed!")
    else:
        print("[MT] No faces extracted!")

    # ========================================
    # Path 2: TSDF Fusion + Evaluation
    # ========================================
    print("\n" + "=" * 60)
    print("[TSDF] Testing TSDF Fusion path...")
    print("=" * 60)

    from scripts.extract_mesh_tsdf import TSDFExtractor, post_process_mesh as tsdf_post_process

    with torch.no_grad():
        tsdf_extractor = TSDFExtractor(
            model,
            depth_ratio=0.0,  # 0.0 = 100% median depth, 1.0 = 100% expected depth
            tile_size=4,
            min_t=0.4
        )

        print("    Rendering depth maps...")
        tsdf_extractor.reconstruction(train_cameras)

        print("    Fusing TSDF...")
        tsdf_mesh = tsdf_extractor.extract_mesh(
            voxel_size=0.002,
            sdf_trunc=0.01,
            depth_trunc=8.0
        )

        verts_np = np.asarray(tsdf_mesh.vertices)
        faces_np = np.asarray(tsdf_mesh.triangles)
        print(f"    Raw TSDF mesh: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # Post-process
        tsdf_mesh = tsdf_post_process(tsdf_mesh, cluster_to_keep=50)
        verts_np = np.asarray(tsdf_mesh.vertices)
        faces_np = np.asarray(tsdf_mesh.triangles)
        print(f"    After post-process: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # Save mesh
        tsdf_mesh_path = OUTPUT_DIR / "mesh_tsdf.ply"
        o3d.io.write_triangle_mesh(str(tsdf_mesh_path), tsdf_mesh)
        print(f"    Saved: {tsdf_mesh_path}")

        # Run DTU evaluation
        print(f"\n    Running DTU evaluation...")
        from eval.dtu.evaluate import evaluate_dtu_mesh

        tsdf_results = evaluate_dtu_mesh(
            mesh_path=str(tsdf_mesh_path),
            train_cameras=train_cameras,
            dtu_path=DTU_ROOT,
            scan_id=SCAN_ID,
            output_dir=str(OUTPUT_DIR / 'eval_tsdf'),
            cull=True,
            dilation_radius=6,
            dataset_path=str(DATASET_PATH),  # 传入 dataset_path 以使用 scale_mat 对齐
        )

        if tsdf_results:
            print(f"\n[TSDF] Results:")
            print(f"    mean_d2s: {tsdf_results['mean_d2s']:.4f}")
            print(f"    mean_s2d: {tsdf_results['mean_s2d']:.4f}")
            print(f"    overall:  {tsdf_results['overall']:.4f}")
        else:
            print("[TSDF] Evaluation failed!")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if mt_results and tsdf_results:
        print(f"MT   overall: {mt_results['overall']:.4f}")
        print(f"TSDF overall: {tsdf_results['overall']:.4f}")
        print("\nTest completed successfully!")
    else:
        print("Some evaluations failed. Check logs above.")


if __name__ == '__main__':
    main()
