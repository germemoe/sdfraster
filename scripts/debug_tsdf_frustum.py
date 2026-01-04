#!/usr/bin/env python3
"""
分析 TSDF 重建范围问题

检查相机可见区域 vs TSDF 重建区域
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render


def compute_visible_frustum(cam, depth_min, depth_max):
    """计算相机在给定深度范围内的可见 frustum 角点"""
    W, H = cam.image_width, cam.image_height
    fx = W / (2 * math.tan(cam.fovx / 2.))
    fy = H / (2 * math.tan(cam.fovy / 2.))
    cx, cy = W / 2., H / 2.

    # 图像角点 -> 归一化相机坐标
    corners_2d = [
        (0, 0),      # 左上
        (W, 0),      # 右上
        (0, H),      # 左下
        (W, H),      # 右下
    ]

    # 相机到世界变换
    w2v = cam.world_view_transform.T.numpy()
    v2w = np.linalg.inv(w2v)

    frustum_points = []
    for px, py in corners_2d:
        # 归一化相机坐标
        x = (px - cx) / fx
        y = (py - cy) / fy
        ray_cam = np.array([x, y, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # 转换到世界坐标
        ray_world = (v2w[:3, :3] @ ray_cam)
        cam_pos = v2w[:3, 3]

        # 近平面和远平面的点
        for d in [depth_min, depth_max]:
            point = cam_pos + ray_world * d
            frustum_points.append(point)

    return np.array(frustum_points)


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

    # 渲染几个相机的深度并统计
    print("\n" + "="*70)
    print("Depth Statistics from Rendering")
    print("="*70)

    all_depth_min = []
    all_depth_max = []
    all_frustum_points = []

    for i in range(0, len(train_cameras), 10):  # 每 10 个相机采样一个
        cam = train_cameras[i]
        with torch.no_grad():
            render_pkg = render(
                cam, model,
                scene_scaling=model.scene_scaling,
                min_t=getattr(config, 'min_t', 0.4),
                tile_size=config.tile_size
            )

        depth = render_pkg['depth_med'].cpu().numpy()
        alpha = render_pkg['alpha'].cpu().numpy()
        valid_mask = alpha > 0.5

        if valid_mask.any():
            valid_depth = depth[valid_mask]
            d_min, d_max = valid_depth.min(), valid_depth.max()
            all_depth_min.append(d_min)
            all_depth_max.append(d_max)
            print(f"Camera {i}: depth range [{d_min:.4f}, {d_max:.4f}]")

            # 计算该相机的 frustum
            frustum = compute_visible_frustum(cam, d_min, d_max)
            all_frustum_points.append(frustum)

    # 总体统计
    print("\n" + "="*70)
    print("Overall Statistics")
    print("="*70)
    all_depth_min = np.array(all_depth_min)
    all_depth_max = np.array(all_depth_max)
    print(f"Depth range across all cameras: [{all_depth_min.min():.4f}, {all_depth_max.max():.4f}]")
    print(f"Average depth: [{all_depth_min.mean():.4f}, {all_depth_max.mean():.4f}]")

    # 合并所有 frustum 点
    all_frustum_points = np.vstack(all_frustum_points)
    print(f"\nCombined frustum bounds (world space):")
    print(f"  X: [{all_frustum_points[:, 0].min():.4f}, {all_frustum_points[:, 0].max():.4f}]")
    print(f"  Y: [{all_frustum_points[:, 1].min():.4f}, {all_frustum_points[:, 1].max():.4f}]")
    print(f"  Z: [{all_frustum_points[:, 2].min():.4f}, {all_frustum_points[:, 2].max():.4f}]")
    print(f"  Size: {all_frustum_points.max(axis=0) - all_frustum_points.min(axis=0)}")

    # 与 TSDF mesh 比较
    import open3d as o3d
    tsdf_mesh = o3d.io.read_triangle_mesh(str(model_path / 'mesh_tsdf.ply'))
    tsdf_verts = np.asarray(tsdf_mesh.vertices)
    print(f"\nTSDF mesh bounds:")
    print(f"  X: [{tsdf_verts[:, 0].min():.4f}, {tsdf_verts[:, 0].max():.4f}]")
    print(f"  Y: [{tsdf_verts[:, 1].min():.4f}, {tsdf_verts[:, 1].max():.4f}]")
    print(f"  Z: [{tsdf_verts[:, 2].min():.4f}, {tsdf_verts[:, 2].max():.4f}]")
    print(f"  Size: {tsdf_verts.max(axis=0) - tsdf_verts.min(axis=0)}")

    print("\n" + "="*70)
    print("TSDF Parameters")
    print("="*70)
    print(f"depth_trunc: {getattr(config, 'tsdf_depth_trunc', 8.0)}")
    print(f"voxel_size: {getattr(config, 'tsdf_voxel_size', 0.002)}")

    # 检查是否 frustum 被 depth_trunc 截断
    depth_trunc = getattr(config, 'tsdf_depth_trunc', 8.0)
    if all_depth_max.max() > depth_trunc:
        print(f"\nWARNING: Max depth {all_depth_max.max():.4f} > depth_trunc {depth_trunc}")
        print("This may cause truncation of the reconstruction!")


if __name__ == "__main__":
    main()
