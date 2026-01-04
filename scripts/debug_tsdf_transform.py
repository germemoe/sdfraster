#!/usr/bin/env python3
"""
分析 TSDF mesh 坐标变换问题

比较不同变换方式的效果
"""

import os
import sys
import numpy as np
from pathlib import Path
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import loader
from utils.args import Args


def main():
    model_path = Path('/home/jk/jk2/project/output/lambda_ablation/scan24/es0_ndm0_md0_mn0')
    config = Args.load_from_json(str(model_path / "config.json"))

    # 加载 scene info
    _, _, scene_info = loader.load_dataset(
        config.dataset_path, config.image_folder,
        data_device='cpu', eval=True, resolution=config.resolution
    )

    # 加载 scale_mat
    cameras_npz = os.path.join(config.dataset_path, 'cameras.npz')
    cam_dict = np.load(cameras_npz)
    scale_mat = cam_dict['scale_mat_0']

    # 加载 TSDF mesh（训练空间）
    tsdf_mesh_path = model_path / 'mesh_tsdf.ply'
    mesh = o3d.io.read_triangle_mesh(str(tsdf_mesh_path))
    verts = np.asarray(mesh.vertices)

    print("="*70)
    print("TSDF Mesh in Training Space")
    print("="*70)
    print(f"Vertices: {len(verts)}")
    print(f"Range: {verts.min(axis=0)} to {verts.max(axis=0)}")
    print(f"Size: {verts.max(axis=0) - verts.min(axis=0)}")

    # 加载 GT mesh
    scan_id = '024'
    gt_path = f'/storage/jk2/dataset/DTU/Points/stl/stl{scan_id}_total.ply'
    gt_mesh = o3d.io.read_triangle_mesh(gt_path)
    gt_verts = np.asarray(gt_mesh.vertices)

    print("\n" + "="*70)
    print("GT Mesh")
    print("="*70)
    print(f"Vertices: {len(gt_verts)}")
    print(f"Range: {gt_verts.min(axis=0)} to {gt_verts.max(axis=0)}")
    print(f"Size: {gt_verts.max(axis=0) - gt_verts.min(axis=0)}")
    print(f"Center: {(gt_verts.max(axis=0) + gt_verts.min(axis=0)) / 2}")

    # 方法 1: 原始变换链 (inv_transform -> scale_mat)
    print("\n" + "="*70)
    print("Method 1: inv(transform) -> scale_mat")
    print("="*70)
    inv_transform = np.linalg.inv(scene_info.transform)
    verts_hom = np.hstack([verts, np.ones((len(verts), 1))])
    verts_colmap = (inv_transform @ verts_hom.T).T[:, :3]
    verts_world1 = (scale_mat @ np.hstack([verts_colmap, np.ones((len(verts_colmap), 1))]).T).T[:, :3]
    print(f"Range: {verts_world1.min(axis=0)} to {verts_world1.max(axis=0)}")
    print(f"Size: {verts_world1.max(axis=0) - verts_world1.min(axis=0)}")
    print(f"Center: {(verts_world1.max(axis=0) + verts_world1.min(axis=0)) / 2}")

    # 方法 2: 直接 scale_mat (跳过 inv_transform)
    print("\n" + "="*70)
    print("Method 2: scale_mat only (skip inv_transform)")
    print("="*70)
    verts_world2 = (scale_mat @ verts_hom.T).T[:, :3]
    print(f"Range: {verts_world2.min(axis=0)} to {verts_world2.max(axis=0)}")
    print(f"Size: {verts_world2.max(axis=0) - verts_world2.min(axis=0)}")
    print(f"Center: {(verts_world2.max(axis=0) + verts_world2.min(axis=0)) / 2}")

    # 方法 3: 组合变换 (scale_mat @ inv_transform)
    print("\n" + "="*70)
    print("Method 3: Combined (scale_mat @ inv_transform)")
    print("="*70)
    combined = scale_mat @ inv_transform
    verts_world3 = (combined @ verts_hom.T).T[:, :3]
    print(f"Range: {verts_world3.min(axis=0)} to {verts_world3.max(axis=0)}")
    print(f"Size: {verts_world3.max(axis=0) - verts_world3.min(axis=0)}")
    print(f"Center: {(verts_world3.max(axis=0) + verts_world3.min(axis=0)) / 2}")

    # 方法 4: 缩放因子分析
    print("\n" + "="*70)
    print("Scale Factor Analysis")
    print("="*70)
    gt_size = gt_verts.max(axis=0) - gt_verts.min(axis=0)
    our_size_train = verts.max(axis=0) - verts.min(axis=0)
    our_size_world = verts_world1.max(axis=0) - verts_world1.min(axis=0)

    print(f"GT size: {gt_size}")
    print(f"Our size (training): {our_size_train}")
    print(f"Our size (world): {our_size_world}")
    print(f"Ratio (GT / our_world): {gt_size / our_size_world}")

    # 方法 5: 检查是否需要额外缩放
    print("\n" + "="*70)
    print("Method 5: With additional scaling")
    print("="*70)
    # 计算需要的额外缩放因子
    scale_needed = gt_size.mean() / our_size_world.mean()
    print(f"Scale factor needed: {scale_needed}")

    # 应用额外缩放
    gt_center = (gt_verts.max(axis=0) + gt_verts.min(axis=0)) / 2
    our_center = (verts_world1.max(axis=0) + verts_world1.min(axis=0)) / 2
    verts_world5 = (verts_world1 - our_center) * scale_needed + gt_center
    print(f"After scaling and centering:")
    print(f"Range: {verts_world5.min(axis=0)} to {verts_world5.max(axis=0)}")
    print(f"Size: {verts_world5.max(axis=0) - verts_world5.min(axis=0)}")
    print(f"Center: {(verts_world5.max(axis=0) + verts_world5.min(axis=0)) / 2}")


if __name__ == "__main__":
    main()
