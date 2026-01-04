#!/usr/bin/env python3
"""
简化版深度对齐测试：检查两个相邻视角是否投影到同一位置

如果 TSDF 产生多面墙，说明深度投影不一致
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


def depth_to_points_full(depth, cam):
    """将深度图完整投影到世界坐标系的3D点"""
    H, W = depth.shape

    # 内参
    fx = W / (2 * math.tan(cam.fovx / 2.))
    fy = H / (2 * math.tan(cam.fovy / 2.))
    cx, cy = W / 2., H / 2.

    # 生成像素坐标
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = u.flatten().astype(np.float64)
    v = v.flatten().astype(np.float64)
    d = depth.flatten()

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

    return points_world, u.astype(np.int32), v.astype(np.int32)


def project_points_to_image(points_world, cam):
    """将世界坐标点投影到图像平面"""
    H = cam.image_height
    W = cam.image_width

    # 内参
    fx = W / (2 * math.tan(cam.fovx / 2.))
    fy = H / (2 * math.tan(cam.fovy / 2.))
    cx, cy = W / 2., H / 2.

    # 世界坐标 -> 相机坐标
    w2c = cam.world_view_transform.T.cpu().numpy()

    points_hom = np.hstack([points_world, np.ones((len(points_world), 1))])
    points_cam = (w2c @ points_hom.T).T[:, :3]

    # 过滤相机后方的点
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]

    # 投影到像素坐标
    u = points_cam[:, 0] / points_cam[:, 2] * fx + cx
    v = points_cam[:, 1] / points_cam[:, 2] * fy + cy
    z = points_cam[:, 2]

    # 过滤图像外的点
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    return u, v, z, valid, in_image


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

    # 选择两个相邻相机
    cam_a_idx, cam_b_idx = 0, 1

    cam_a = train_cameras[cam_a_idx]
    cam_b = train_cameras[cam_b_idx]

    print(f"\n{'='*70}")
    print(f"Testing depth consistency between Camera {cam_a_idx} and Camera {cam_b_idx}")
    print(f"{'='*70}")

    # 渲染两个相机的深度
    with torch.no_grad():
        render_a = render(
            cam_a, model,
            scene_scaling=model.scene_scaling,
            min_t=getattr(config, 'min_t', 0.4),
            tile_size=config.tile_size
        )
        render_b = render(
            cam_b, model,
            scene_scaling=model.scene_scaling,
            min_t=getattr(config, 'min_t', 0.4),
            tile_size=config.tile_size
        )

    depth_a = render_a['depth_med'].cpu().numpy()
    depth_b = render_b['depth_med'].cpu().numpy()

    # 投影 Camera A 的深度到 3D
    points_a, u_a, v_a = depth_to_points_full(depth_a, cam_a)
    print(f"\nCamera {cam_a_idx}: {len(points_a)} valid 3D points")

    # 将 Camera A 的 3D 点投影到 Camera B
    u_proj, v_proj, z_proj, valid_front, in_image = project_points_to_image(points_a, cam_b)

    # 只取在图像内的点
    u_proj_int = u_proj[in_image].astype(np.int32)
    v_proj_int = v_proj[in_image].astype(np.int32)
    z_proj_valid = z_proj[in_image]

    # 获取 Camera B 在这些像素的深度
    depth_b_at_proj = depth_b[v_proj_int, u_proj_int]

    # 比较深度
    depth_diff = np.abs(z_proj_valid - depth_b_at_proj)

    print(f"\nProjected {np.sum(in_image)} points from Camera {cam_a_idx} to Camera {cam_b_idx}")
    print(f"Depth from Camera {cam_a_idx} projection: {z_proj_valid.min():.4f} to {z_proj_valid.max():.4f}")
    print(f"Depth from Camera {cam_b_idx} render: {depth_b_at_proj.min():.4f} to {depth_b_at_proj.max():.4f}")

    print(f"\nDepth difference statistics:")
    print(f"  Min:    {depth_diff.min():.6f}")
    print(f"  Max:    {depth_diff.max():.6f}")
    print(f"  Mean:   {depth_diff.mean():.6f}")
    print(f"  Median: {np.median(depth_diff):.6f}")
    print(f"  Std:    {depth_diff.std():.6f}")

    # 计算有多少点的深度差异超过阈值
    thresholds = [0.01, 0.02, 0.05, 0.1]
    for thresh in thresholds:
        pct = np.mean(depth_diff > thresh) * 100
        print(f"  > {thresh}: {pct:.2f}%")

    # 这个差异应该很小（<0.01），如果很大说明投影有问题
    if depth_diff.mean() > 0.05:
        print("\nWARNING: Large depth inconsistency detected!")
        print("This may cause multi-surface artifacts in TSDF fusion.")

    # 检查 TSDF 参数是否合理
    print(f"\n{'='*70}")
    print("TSDF parameter analysis")
    print(f"{'='*70}")

    voxel_size = 0.002
    sdf_trunc = 0.01
    depth_trunc = 8.0

    print(f"voxel_size: {voxel_size}")
    print(f"sdf_trunc: {sdf_trunc}")
    print(f"depth_trunc: {depth_trunc}")
    print(f"sdf_trunc / voxel_size: {sdf_trunc / voxel_size}")

    # 深度差异相对于 sdf_trunc 的比例
    print(f"\nDepth difference relative to sdf_trunc ({sdf_trunc}):")
    print(f"  Mean diff / sdf_trunc: {depth_diff.mean() / sdf_trunc:.2f}")
    print(f"  Max diff / sdf_trunc: {depth_diff.max() / sdf_trunc:.2f}")

    if depth_diff.mean() > sdf_trunc:
        print("\nWARNING: Mean depth inconsistency exceeds sdf_trunc!")
        print("This will definitely cause multi-surface artifacts.")

    # 保存深度差异可视化
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Depth A
    axes[0, 0].imshow(depth_a, cmap='viridis')
    axes[0, 0].set_title(f'Camera {cam_a_idx} Depth')
    axes[0, 0].axis('off')

    # Depth B
    axes[0, 1].imshow(depth_b, cmap='viridis')
    axes[0, 1].set_title(f'Camera {cam_b_idx} Depth')
    axes[0, 1].axis('off')

    # Depth difference map
    diff_map = np.zeros_like(depth_b)
    diff_map[v_proj_int, u_proj_int] = depth_diff
    axes[0, 2].imshow(diff_map, cmap='hot', vmin=0, vmax=0.1)
    axes[0, 2].set_title('Depth Difference')
    axes[0, 2].axis('off')

    # Histogram
    axes[1, 0].hist(depth_diff, bins=100, range=(0, 0.2))
    axes[1, 0].set_xlabel('Depth Difference')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Depth Difference Distribution')
    axes[1, 0].axvline(x=sdf_trunc, color='r', linestyle='--', label=f'sdf_trunc={sdf_trunc}')
    axes[1, 0].legend()

    # Points projected from A to B
    proj_vis = np.zeros((cam_b.image_height, cam_b.image_width))
    proj_vis[v_proj_int, u_proj_int] = 1
    axes[1, 1].imshow(proj_vis, cmap='gray')
    axes[1, 1].set_title('Projection Coverage')
    axes[1, 1].axis('off')

    # 深度 A 投影 vs 深度 B
    axes[1, 2].scatter(z_proj_valid[::100], depth_b_at_proj[::100], alpha=0.3, s=1)
    axes[1, 2].plot([1, 2], [1, 2], 'r--')
    axes[1, 2].set_xlabel('Projected Depth from A')
    axes[1, 2].set_ylabel('Rendered Depth from B')
    axes[1, 2].set_title('Depth Consistency')
    axes[1, 2].set_aspect('equal')

    plt.tight_layout()
    output_path = model_path / 'debug_depth_viz' / 'depth_consistency.png'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
