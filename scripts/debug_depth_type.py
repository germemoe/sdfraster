#!/usr/bin/env python3
"""
检查深度是 t-depth 还是 z-depth

t-depth: 沿射线的距离
z-depth: 沿相机 z 轴的距离

如果渲染输出的是 t-depth，而 Open3D 期望 z-depth，就会造成投影不一致
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

    cam = train_cameras[0]
    H = cam.image_height
    W = cam.image_width

    # 渲染
    with torch.no_grad():
        render_pkg = render(
            cam, model,
            scene_scaling=model.scene_scaling,
            min_t=getattr(config, 'min_t', 0.4),
            tile_size=config.tile_size
        )

    depth_med = render_pkg['depth_med'].cpu().numpy()
    depth_exp = render_pkg['depth'].cpu().numpy()

    print(f"\n{'='*70}")
    print("Depth Type Analysis")
    print(f"{'='*70}")
    print(f"depth_med range: [{depth_med.min():.4f}, {depth_med.max():.4f}]")
    print(f"depth_exp range: [{depth_exp.min():.4f}, {depth_exp.max():.4f}]")

    # 计算每个像素的射线方向
    fx = W / (2 * math.tan(cam.fovx / 2.))
    fy = H / (2 * math.tan(cam.fovy / 2.))
    cx, cy = W / 2., H / 2.

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # 相机坐标系下的射线方向
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(u, dtype=np.float64)

    # 射线方向的模长
    ray_len = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)

    # 如果 depth 是 t-depth，那么 z-depth = depth / ray_len
    # 如果 depth 是 z-depth，那么 t-depth = depth * ray_len

    # 检查中心像素（射线方向 = (0, 0, 1)，此时 t-depth = z-depth）
    center_u, center_v = W // 2, H // 2
    print(f"\nCenter pixel ({center_u}, {center_v}):")
    print(f"  depth_med = {depth_med[center_v, center_u]:.6f}")
    print(f"  ray_len = {ray_len[center_v, center_u]:.6f} (should be 1.0)")

    # 检查角落像素（射线方向偏离 z 轴最多）
    corner_u, corner_v = 0, 0
    print(f"\nCorner pixel ({corner_u}, {corner_v}):")
    print(f"  depth_med = {depth_med[corner_v, corner_u]:.6f}")
    print(f"  ray_len = {ray_len[corner_v, corner_u]:.6f}")

    # 如果深度是 z-depth，那么 3D 点的 z 坐标应该等于 depth
    # 如果深度是 t-depth，那么 3D 点的 z 坐标应该等于 depth / ray_len

    # 对角落像素进行验证
    d = depth_med[corner_v, corner_u]
    if d > 0:
        # 假设是 z-depth：点的位置
        x_3d_z = x_cam[corner_v, corner_u] * d
        y_3d_z = y_cam[corner_v, corner_u] * d
        z_3d_z = d

        # 假设是 t-depth：点的位置
        t = d
        x_3d_t = x_cam[corner_v, corner_u] * t / ray_len[corner_v, corner_u]
        y_3d_t = y_cam[corner_v, corner_u] * t / ray_len[corner_v, corner_u]
        z_3d_t = t / ray_len[corner_v, corner_u]

        print(f"\n  If z-depth: point = ({x_3d_z:.4f}, {y_3d_z:.4f}, {z_3d_z:.4f})")
        print(f"  If t-depth: point = ({x_3d_t:.4f}, {y_3d_t:.4f}, {z_3d_t:.4f})")

    # 检查深度和射线长度的关系
    # 如果是 z-depth，depth 和 ray_len 应该没有明显相关性
    # 如果是 t-depth，depth * cos(theta) 应该给出 z-depth

    # 计算 cos(theta) = 1 / ray_len
    cos_theta = 1.0 / ray_len

    # 比较边缘和中心的深度
    print(f"\n{'='*70}")
    print("Depth at different image locations")
    print(f"{'='*70}")

    locations = [
        (W // 2, H // 2, "center"),
        (0, 0, "top-left"),
        (W - 1, 0, "top-right"),
        (0, H - 1, "bottom-left"),
        (W - 1, H - 1, "bottom-right"),
    ]

    for u, v, name in locations:
        d = depth_med[v, u]
        rl = ray_len[v, u]
        z_if_t = d / rl if d > 0 else 0
        print(f"{name:12s}: depth={d:.4f}, ray_len={rl:.4f}, z_if_t={z_if_t:.4f}")

    # 关键测试：投影一个点并检查深度
    print(f"\n{'='*70}")
    print("Reprojection Test")
    print(f"{'='*70}")

    # 取一个非中心像素
    test_u, test_v = W // 4, H // 4
    d = depth_med[test_v, test_u]

    if d > 0:
        # 方法 1: 假设 z-depth
        x_cam_1 = (test_u - cx) / fx * d
        y_cam_1 = (test_v - cy) / fy * d
        z_cam_1 = d
        point_cam_1 = np.array([x_cam_1, y_cam_1, z_cam_1])

        # 投影回像素
        u_reproj_1 = point_cam_1[0] / point_cam_1[2] * fx + cx
        v_reproj_1 = point_cam_1[1] / point_cam_1[2] * fy + cy

        print(f"Method 1 (assume z-depth):")
        print(f"  Original pixel: ({test_u}, {test_v})")
        print(f"  Reprojected pixel: ({u_reproj_1:.2f}, {v_reproj_1:.2f})")
        print(f"  Error: ({abs(u_reproj_1 - test_u):.4f}, {abs(v_reproj_1 - test_v):.4f})")

        # 方法 2: 假设 t-depth，转换为 z-depth
        rl = ray_len[test_v, test_u]
        z_from_t = d / rl
        x_cam_2 = (test_u - cx) / fx * z_from_t
        y_cam_2 = (test_v - cy) / fy * z_from_t
        z_cam_2 = z_from_t
        point_cam_2 = np.array([x_cam_2, y_cam_2, z_cam_2])

        # 投影回像素
        u_reproj_2 = point_cam_2[0] / point_cam_2[2] * fx + cx
        v_reproj_2 = point_cam_2[1] / point_cam_2[2] * fy + cy

        print(f"\nMethod 2 (assume t-depth, convert to z-depth):")
        print(f"  Original pixel: ({test_u}, {test_v})")
        print(f"  Reprojected pixel: ({u_reproj_2:.2f}, {v_reproj_2:.2f})")
        print(f"  Error: ({abs(u_reproj_2 - test_u):.4f}, {abs(v_reproj_2 - test_v):.4f})")

    # 结论
    print(f"\n{'='*70}")
    print("Conclusion")
    print(f"{'='*70}")
    print("If Method 1 has small error -> depth is z-depth (correct for Open3D)")
    print("If Method 2 has small error -> depth is t-depth (needs conversion)")


if __name__ == "__main__":
    main()
