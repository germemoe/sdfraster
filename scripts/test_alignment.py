#!/usr/bin/env python3
"""
测试坐标对齐是否正确的脚本。
验证 evaluate.py 中的 align_mesh_with_transforms() 函数。
"""

import numpy as np
import sys
sys.path.insert(0, '/data/jk/project/radmesh')

from eval.dtu.evaluate import (
    load_pca_transform,
    load_scale_mat,
    align_mesh_with_transforms
)

# Paths
MODEL_OUTPUT_PATH = "/home/jk/data/project/radmesh/newoutput/sdf/DTU/test/just_tv1e-3_normwdmed_meshdmed_hash22_scan65_fixext2"
DATASET_PATH = "/storage/jk/dataset/DTU/scan65"
MESH_PATH = "/data/jk/project/radmesh/test_eval_output/mesh_mt.ply"
GT_POINTS_PATH = "/storage/jk/dataset/DTU/Points/stl/stl065_total.ply"

print("=" * 60)
print("Testing align_mesh_with_transforms()")
print("=" * 60)

# 1. Load transforms
print("\n[1] Loading transforms...")
pca_transform = load_pca_transform(MODEL_OUTPUT_PATH)
scale_mat = load_scale_mat(DATASET_PATH)

if pca_transform is not None:
    print(f"    PCA transform loaded (diag: {np.diag(pca_transform)[:3]})")
else:
    print("    PCA transform: None")

if scale_mat is not None:
    print(f"    scale_mat loaded (diag: {np.diag(scale_mat)[:3]})")
else:
    print("    scale_mat: None")

# 2. Load mesh
print("\n[2] Loading mesh...")
import open3d as o3d
mesh = o3d.io.read_triangle_mesh(MESH_PATH)
vertices = np.asarray(mesh.vertices)
print(f"    Vertices: {len(vertices)}")
print(f"    Range Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")

# 3. Apply alignment
print("\n[3] Applying align_mesh_with_transforms()...")
aligned_verts = align_mesh_with_transforms(vertices, pca_transform, scale_mat)
print(f"    Range X: [{aligned_verts[:, 0].min():.2f}, {aligned_verts[:, 0].max():.2f}]")
print(f"    Range Y: [{aligned_verts[:, 1].min():.2f}, {aligned_verts[:, 1].max():.2f}]")
print(f"    Range Z: [{aligned_verts[:, 2].min():.2f}, {aligned_verts[:, 2].max():.2f}]")

# 4. Compare with GT
print("\n[4] Comparing with GT point cloud...")
gt_pcd = o3d.io.read_point_cloud(GT_POINTS_PATH)
gt_points = np.asarray(gt_pcd.points)
print(f"    GT Range X: [{gt_points[:, 0].min():.2f}, {gt_points[:, 0].max():.2f}]")
print(f"    GT Range Y: [{gt_points[:, 1].min():.2f}, {gt_points[:, 1].max():.2f}]")
print(f"    GT Range Z: [{gt_points[:, 2].min():.2f}, {gt_points[:, 2].max():.2f}]")

# 5. Center distance
mesh_center = aligned_verts.mean(axis=0)
gt_center = gt_points.mean(axis=0)
center_diff = np.linalg.norm(mesh_center - gt_center)
print(f"\n[5] Alignment check:")
print(f"    Mesh center: [{mesh_center[0]:.2f}, {mesh_center[1]:.2f}, {mesh_center[2]:.2f}]")
print(f"    GT center:   [{gt_center[0]:.2f}, {gt_center[1]:.2f}, {gt_center[2]:.2f}]")
print(f"    Center distance: {center_diff:.2f}")

if center_diff < 100:
    print(f"    ✓ Alignment looks correct!")
else:
    print(f"    ✗ Alignment may be wrong!")

print("\n" + "=" * 60)
