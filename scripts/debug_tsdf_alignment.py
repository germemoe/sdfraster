#!/usr/bin/env python3
"""
诊断 TSDF 多面墙问题：检查各视角深度投影的对齐情况

将多个视角的深度图投影到3D点云，可视化检查是否存在系统性偏移
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render


def depth_to_points(depth, cam, subsample=10):
    """将深度图投影到世界坐标系的3D点"""
    H, W = depth.shape

    # 内参
    fx = W / (2 * math.tan(cam.fovx / 2.))
    fy = H / (2 * math.tan(cam.fovy / 2.))
    cx, cy = W / 2., H / 2.

    # 生成像素坐标
    v, u = np.meshgrid(np.arange(0, H, subsample), np.arange(0, W, subsample), indexing='ij')
    u = u.flatten()
    v = v.flatten()
    d = depth[v, u]

    # 过滤无效深度
    valid = (d > 0) & (d < 100)
    u, v, d = u[valid], v[valid], d[valid]

    # 像素坐标 -> 相机坐标
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # 相机坐标 -> 世界坐标
    w2c = cam.world_view_transform.T.cpu().numpy()
    c2w = np.linalg.inv(w2c)

    points_cam_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    points_world = (c2w @ points_cam_hom.T).T[:, :3]

    return points_world


def main():
    model_path = Path('/home/jk/jk2/project/output/lambda_ablation/scan24/es0_ndm0_md0_mn0')
    config = Args.load_from_json(str(model_path / "config.json"))

    # 加载数据集
    print(f"Loading dataset...")
    train_cameras, _, scene_info = loader.load_dataset(
        config.dataset_path, config.image_folder,
        data_device='cpu', eval=True, resolution=config.resolution
    )

    # 加载模型
    print(f"Loading model...")
    device = torch.device('cuda')
    model = Model.load_ckpt(model_path, device)
    model.eval()

    # 选择几个相邻视角
    # 这些视角应该看到同一面墙的不同部分
    cam_indices = [0, 1, 2, 3, 4]  # 连续的相机更容易比较

    colors = [
        [1, 0, 0],  # 红
        [0, 1, 0],  # 绿
        [0, 0, 1],  # 蓝
        [1, 1, 0],  # 黄
        [1, 0, 1],  # 紫
    ]

    all_points = []
    all_colors = []
    camera_positions = []

    print("\n" + "="*70)
    print("Projecting depth to 3D points for each camera")
    print("="*70)

    for i, cam_idx in enumerate(cam_indices):
        cam = train_cameras[cam_idx]

        with torch.no_grad():
            render_pkg = render(
                cam, model,
                scene_scaling=model.scene_scaling,
                min_t=getattr(config, 'min_t', 0.4),
                tile_size=config.tile_size
            )

        depth = render_pkg['depth_med'].cpu().numpy()

        # 投影到3D
        points = depth_to_points(depth, cam, subsample=20)

        # 记录相机位置
        w2c = cam.world_view_transform.T.cpu().numpy()
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        camera_positions.append(cam_pos)

        print(f"Camera {cam_idx}: {len(points)} points, cam_pos = {cam_pos}")
        print(f"  Point range: {points.min(axis=0)} to {points.max(axis=0)}")

        all_points.append(points)
        all_colors.append(np.tile(colors[i], (len(points), 1)))

    # 合并所有点
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # 保存点云
    output_path = model_path / 'debug_depth_viz' / 'depth_alignment.ply'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"\nSaved point cloud to {output_path}")

    # 分析重叠区域
    print("\n" + "="*70)
    print("Analyzing overlap regions")
    print("="*70)

    # 用 voxel grid 找到重叠区域
    voxel_size = 0.01

    # 对每个视角建立 voxel grid
    voxel_grids = []
    for i, points in enumerate([depth_to_points(
        render(train_cameras[cam_idx], model,
               scene_scaling=model.scene_scaling,
               min_t=getattr(config, 'min_t', 0.4),
               tile_size=config.tile_size)['depth_med'].cpu().numpy(),
        train_cameras[cam_idx], subsample=5) for cam_idx in cam_indices[:3]]):

        # 量化到 voxel
        voxels = np.floor(points / voxel_size).astype(np.int64)
        voxel_set = set(map(tuple, voxels))
        voxel_grids.append(voxel_set)
        print(f"Camera {cam_indices[i]}: {len(voxel_set)} unique voxels")

    # 找交集
    if len(voxel_grids) >= 2:
        overlap_01 = voxel_grids[0] & voxel_grids[1]
        print(f"\nOverlap between cam {cam_indices[0]} and {cam_indices[1]}: {len(overlap_01)} voxels")

        if len(voxel_grids) >= 3:
            overlap_012 = overlap_01 & voxel_grids[2]
            print(f"Overlap among cam {cam_indices[0]}, {cam_indices[1]}, {cam_indices[2]}: {len(overlap_012)} voxels")

    # 检查 scene_scaling 的影响
    print("\n" + "="*70)
    print("scene_scaling analysis")
    print("="*70)
    print(f"scene_scaling = {model.scene_scaling}")

    # 渲染一个视角，比较有无 scene_scaling 的区别
    cam = train_cameras[0]
    with torch.no_grad():
        render_pkg = render(
            cam, model,
            scene_scaling=model.scene_scaling,
            min_t=getattr(config, 'min_t', 0.4),
            tile_size=config.tile_size
        )

    depth = render_pkg['depth_med'].cpu().numpy()

    # 1. 原始投影
    points_orig = depth_to_points(depth, cam, subsample=20)

    # 2. 除以 scene_scaling
    points_scaled = points_orig / model.scene_scaling

    print(f"Original points range: {points_orig.min(axis=0)} to {points_orig.max(axis=0)}")
    print(f"Scaled points range: {points_scaled.min(axis=0)} to {points_scaled.max(axis=0)}")

    # 加载 GT mesh 比较
    cameras_npz = os.path.join(config.dataset_path, 'cameras.npz')
    if os.path.exists(cameras_npz):
        cam_dict = np.load(cameras_npz)
        scale_mat = cam_dict['scale_mat_0']

        # 应用完整的坐标变换
        inv_transform = np.linalg.inv(scene_info.transform)

        # 原始点 -> COLMAP 空间
        points_hom = np.hstack([points_orig, np.ones((len(points_orig), 1))])
        points_colmap = (inv_transform @ points_hom.T).T[:, :3]

        # COLMAP 空间 -> 世界空间
        points_hom = np.hstack([points_colmap, np.ones((len(points_colmap), 1))])
        points_world = (scale_mat @ points_hom.T).T[:, :3]

        print(f"\nAfter full transform (inv_transform -> scale_mat):")
        print(f"World points range: {points_world.min(axis=0)} to {points_world.max(axis=0)}")

        # 加载 GT
        scan_id = '024'
        gt_path = f'/storage/jk2/dataset/DTU/Points/stl/stl{scan_id}_total.ply'
        if os.path.exists(gt_path):
            gt_mesh = o3d.io.read_triangle_mesh(gt_path)
            gt_verts = np.asarray(gt_mesh.vertices)
            print(f"\nGT mesh range: {gt_verts.min(axis=0)} to {gt_verts.max(axis=0)}")


if __name__ == "__main__":
    main()
