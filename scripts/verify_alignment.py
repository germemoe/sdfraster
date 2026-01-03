#!/usr/bin/env python3
"""
验证mesh对齐是否正确的简单脚本。

通过比较对齐后mesh的顶点范围和DTU GT点云的范围来验证。
"""

import numpy as np
import open3d as o3d
import os

# Paths
MESH_PATH = "/data/jk/project/radmesh/test_eval_output/mesh_mt.ply"
TRANSFORM_PATH = "/home/jk/data/project/radmesh/newoutput/sdf/DTU/test/just_tv1e-3_normwdmed_meshdmed_hash22_scan65_fixext2/transform.txt"
CAMERAS_NPZ_PATH = "/storage/jk/dataset/DTU/scan65/cameras.npz"
GT_POINTS_PATH = "/storage/jk/dataset/DTU/Points/stl/stl065_total.ply"

print("=" * 60)
print("Coordinate Alignment Verification")
print("=" * 60)

# 1. Load mesh (in training coordinates)
mesh = o3d.io.read_triangle_mesh(MESH_PATH)
vertices = np.asarray(mesh.vertices)
print(f"\n[1] Original mesh (training coords):")
print(f"    Range X: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
print(f"    Range Y: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
print(f"    Range Z: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")
print(f"    Center:  [{vertices.mean(axis=0)}]")

# 2. Load PCA transform and invert
pca_transform = np.loadtxt(TRANSFORM_PATH)
pca_transform_inv = np.linalg.inv(pca_transform)
print(f"\n[2] PCA transform (training -> COLMAP/normalized):")
print(f"    {pca_transform}")

# Apply inverse PCA transform
vertices_hom = np.hstack([vertices, np.ones((len(vertices), 1))])
vertices_colmap = (pca_transform_inv @ vertices_hom.T).T[:, :3]
print(f"\n[3] After inv(PCA transform) (COLMAP/normalized coords):")
print(f"    Range X: [{vertices_colmap[:, 0].min():.4f}, {vertices_colmap[:, 0].max():.4f}]")
print(f"    Range Y: [{vertices_colmap[:, 1].min():.4f}, {vertices_colmap[:, 1].max():.4f}]")
print(f"    Range Z: [{vertices_colmap[:, 2].min():.4f}, {vertices_colmap[:, 2].max():.4f}]")
print(f"    Center:  [{vertices_colmap.mean(axis=0)}]")

# 3. Load scale_mat_inv from cameras.npz
cameras_data = np.load(CAMERAS_NPZ_PATH)
if 'scale_mat_inv_0' in cameras_data:
    scale_mat_inv = cameras_data['scale_mat_inv_0']
else:
    scale_mat_inv = np.linalg.inv(cameras_data['scale_mat_0'])
print(f"\n[4] scale_mat_inv (normalized -> DTU world):")
print(f"    {scale_mat_inv}")

# Apply scale_mat_inv
vertices_colmap_hom = np.hstack([vertices_colmap, np.ones((len(vertices_colmap), 1))])
vertices_dtu = (scale_mat_inv @ vertices_colmap_hom.T).T[:, :3]
print(f"\n[5] After scale_mat_inv (DTU world coords):")
print(f"    Range X: [{vertices_dtu[:, 0].min():.4f}, {vertices_dtu[:, 0].max():.4f}]")
print(f"    Range Y: [{vertices_dtu[:, 1].min():.4f}, {vertices_dtu[:, 1].max():.4f}]")
print(f"    Range Z: [{vertices_dtu[:, 2].min():.4f}, {vertices_dtu[:, 2].max():.4f}]")
print(f"    Center:  [{vertices_dtu.mean(axis=0)}]")

# 4. Load GT point cloud and compare
gt_pcd = o3d.io.read_point_cloud(GT_POINTS_PATH)
gt_points = np.asarray(gt_pcd.points)
print(f"\n[6] GT point cloud (DTU world coords):")
print(f"    Range X: [{gt_points[:, 0].min():.4f}, {gt_points[:, 0].max():.4f}]")
print(f"    Range Y: [{gt_points[:, 1].min():.4f}, {gt_points[:, 1].max():.4f}]")
print(f"    Range Z: [{gt_points[:, 2].min():.4f}, {gt_points[:, 2].max():.4f}]")
print(f"    Center:  [{gt_points.mean(axis=0)}]")

# 5. Compare centers
mesh_center = vertices_dtu.mean(axis=0)
gt_center = gt_points.mean(axis=0)
center_diff = np.linalg.norm(mesh_center - gt_center)
print(f"\n[7] Alignment check:")
print(f"    Mesh center (DTU):  {mesh_center}")
print(f"    GT center:          {gt_center}")
print(f"    Center distance:    {center_diff:.4f}")
if center_diff < 50:  # DTU units are roughly in mm
    print(f"    ✓ Centers are close - alignment looks correct!")
else:
    print(f"    ✗ Centers are far apart - alignment may be wrong!")

# 6. Also test: only scale_mat_inv (without PCA inverse) - this is what old code did
vertices_hom_direct = np.hstack([vertices, np.ones((len(vertices), 1))])
vertices_dtu_wrong = (scale_mat_inv @ vertices_hom_direct.T).T[:, :3]
print(f"\n[8] WRONG alignment (only scale_mat_inv, no PCA inverse):")
print(f"    Range X: [{vertices_dtu_wrong[:, 0].min():.4f}, {vertices_dtu_wrong[:, 0].max():.4f}]")
print(f"    Range Y: [{vertices_dtu_wrong[:, 1].min():.4f}, {vertices_dtu_wrong[:, 1].max():.4f}]")
print(f"    Range Z: [{vertices_dtu_wrong[:, 2].min():.4f}, {vertices_dtu_wrong[:, 2].max():.4f}]")
wrong_center = vertices_dtu_wrong.mean(axis=0)
wrong_center_diff = np.linalg.norm(wrong_center - gt_center)
print(f"    Center distance to GT: {wrong_center_diff:.4f}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Correct alignment (with PCA inverse): center dist = {center_diff:.4f}")
print(f"Wrong alignment (without PCA inverse): center dist = {wrong_center_diff:.4f}")
if center_diff < wrong_center_diff:
    print("→ PCA inverse is needed for correct alignment!")
else:
    print("→ Something unexpected - check the transforms")
