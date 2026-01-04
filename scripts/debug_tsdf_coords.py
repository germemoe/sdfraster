#!/usr/bin/env python3
"""
诊断 TSDF 坐标变换问题的脚本

不修改任何现有代码，只做分析和可视化。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math
from argparse import ArgumentParser
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args


def main():
    parser = ArgumentParser(description="Debug TSDF coordinate transforms")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    config = Args.load_from_json(str(model_path / "config.json"))

    # 加载数据集
    print(f"Loading dataset from {config.dataset_path}...")
    train_cameras, test_cameras, scene_info = loader.load_dataset(
        config.dataset_path, config.image_folder,
        data_device='cpu', eval=True, resolution=config.resolution
    )

    # 加载模型
    print(f"Loading model from {model_path}...")
    device = torch.device('cuda')
    model = Model.load_ckpt(model_path, device)

    print("\n" + "="*70)
    print("1. MODEL SPACE ANALYSIS")
    print("="*70)

    # 模型顶点
    verts = model.vertices.detach().cpu().numpy()
    print(f"Model vertices: {verts.shape}")
    print(f"  Range: [{verts.min(axis=0)}] to [{verts.max(axis=0)}]")
    print(f"  Center: {(verts.max(axis=0) + verts.min(axis=0)) / 2}")
    print(f"  Size: {verts.max(axis=0) - verts.min(axis=0)}")

    print(f"\nModel center: {model.center.cpu().numpy()}")
    print(f"Model scene_scaling: {model.scene_scaling.item()}")

    print("\n" + "="*70)
    print("2. CAMERA ANALYSIS (Training Space)")
    print("="*70)

    # 相机位置（训练空间）
    cam_positions = []
    for cam in train_cameras:
        pos = cam.camera_center.numpy()
        cam_positions.append(pos)
    cam_positions = np.array(cam_positions)
    print(f"Camera positions: {cam_positions.shape}")
    print(f"  Range: [{cam_positions.min(axis=0)}] to [{cam_positions.max(axis=0)}]")
    print(f"  Center: {cam_positions.mean(axis=0)}")

    # 相机朝向（取第一个相机为例）
    cam0 = train_cameras[0]
    w2v = cam0.world_view_transform.T.numpy()  # 4x4
    print(f"\nCamera 0 world_view_transform (W2V):")
    print(w2v)

    # V2W = inv(W2V)
    v2w = np.linalg.inv(w2v)
    print(f"\nCamera 0 V2W (camera to world):")
    print(v2w)

    print("\n" + "="*70)
    print("3. TRANSFORM CHAIN ANALYSIS")
    print("="*70)

    print(f"\nscene_info.transform (PCA alignment):")
    print(scene_info.transform)

    # 逆变换
    inv_transform = np.linalg.inv(scene_info.transform)
    print(f"\ninv(transform):")
    print(inv_transform)

    # scale_mat
    cameras_npz = os.path.join(config.dataset_path, 'cameras.npz')
    if os.path.exists(cameras_npz):
        cam_dict = np.load(cameras_npz)
        scale_mat = cam_dict['scale_mat_0']
        print(f"\nscale_mat_0:")
        print(scale_mat)

        # 提取缩放因子
        scale_factor = scale_mat[0, 0]
        print(f"\nScale factor from scale_mat: {scale_factor}")
    else:
        scale_mat = None
        print("\nNo scale_mat found (not DTU dataset)")

    print("\n" + "="*70)
    print("4. COORDINATE TRANSFORM TEST")
    print("="*70)

    # 测试几个关键点的变换
    test_points = {
        "origin": np.array([0, 0, 0]),
        "model_center": model.center.cpu().numpy().flatten(),
        "cam0_position": cam_positions[0],
        "vertex_center": (verts.max(axis=0) + verts.min(axis=0)) / 2,
    }

    for name, pt in test_points.items():
        pt_hom = np.append(pt, 1)

        # Step 1: Apply inv(transform)
        pt_colmap = (inv_transform @ pt_hom)[:3]

        # Step 2: Apply scale_mat (if exists)
        if scale_mat is not None:
            pt_world = (scale_mat @ np.append(pt_colmap, 1))[:3]
        else:
            pt_world = pt_colmap

        print(f"\n{name}:")
        print(f"  Training space: {pt}")
        print(f"  COLMAP space:   {pt_colmap}")
        print(f"  World space:    {pt_world}")

    print("\n" + "="*70)
    print("5. GT MESH ANALYSIS")
    print("="*70)

    # 加载 GT mesh
    scan_id = ''.join([c for c in os.path.basename(config.dataset_path) if c.isdigit()])
    gt_path = f'/storage/jk2/dataset/DTU/Points/stl/stl{scan_id.zfill(3)}_total.ply'
    if os.path.exists(gt_path):
        gt_mesh = o3d.io.read_triangle_mesh(gt_path)
        gt_verts = np.asarray(gt_mesh.vertices)
        print(f"GT mesh: {gt_path}")
        print(f"  Vertices: {len(gt_verts)}")
        print(f"  Range: [{gt_verts.min(axis=0)}] to [{gt_verts.max(axis=0)}]")
        print(f"  Center: {(gt_verts.max(axis=0) + gt_verts.min(axis=0)) / 2}")
        print(f"  Size: {gt_verts.max(axis=0) - gt_verts.min(axis=0)}")
    else:
        print(f"GT mesh not found at {gt_path}")

    print("\n" + "="*70)
    print("6. TSDF MESH ANALYSIS")
    print("="*70)

    # 检查现有的 TSDF mesh
    tsdf_meshes = list(model_path.glob("mesh_tsdf*.ply"))
    for mesh_path in sorted(tsdf_meshes)[:3]:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh_verts = np.asarray(mesh.vertices)
        if len(mesh_verts) > 0:
            print(f"\n{mesh_path.name}:")
            print(f"  Vertices: {len(mesh_verts)}")
            print(f"  Range: [{mesh_verts.min(axis=0)}] to [{mesh_verts.max(axis=0)}]")
            print(f"  Center: {(mesh_verts.max(axis=0) + mesh_verts.min(axis=0)) / 2}")
            print(f"  Size: {mesh_verts.max(axis=0) - mesh_verts.min(axis=0)}")

    print("\n" + "="*70)
    print("7. DEPTH VALUE ANALYSIS")
    print("="*70)

    # 分析渲染深度与期望距离的关系
    # 相机到模型中心的距离
    model_center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    cam_to_center_dists = np.linalg.norm(cam_positions - model_center, axis=1)
    print(f"Distance from cameras to model center:")
    print(f"  Min: {cam_to_center_dists.min():.4f}")
    print(f"  Max: {cam_to_center_dists.max():.4f}")
    print(f"  Mean: {cam_to_center_dists.mean():.4f}")

    # 如果 GT 存在，计算 GT 尺度下的期望深度
    if scale_mat is not None and os.path.exists(gt_path):
        gt_center = (gt_verts.max(axis=0) + gt_verts.min(axis=0)) / 2
        print(f"\nExpected depth in world space (to GT center):")
        # 相机位置变换到世界空间
        cam0_world = (scale_mat @ np.append((inv_transform @ np.append(cam_positions[0], 1))[:3], 1))[:3]
        dist_world = np.linalg.norm(cam0_world - gt_center)
        print(f"  Cam0 in world space: {cam0_world}")
        print(f"  GT center: {gt_center}")
        print(f"  Distance: {dist_world:.2f}")


if __name__ == "__main__":
    main()
