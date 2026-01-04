#!/usr/bin/env python3
"""
深入分析深度不一致的原因

可能的原因:
1. 遮挡 - 不同视角看到不同表面
2. 射线方向不一致 - 渲染用的射线 vs 投影用的射线
3. scene_scaling 没有正确应用
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt

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

    cam_a = train_cameras[0]
    cam_b = train_cameras[1]

    H = cam_a.image_height
    W = cam_a.image_width

    print(f"\n{'='*70}")
    print("Camera Parameters Comparison")
    print(f"{'='*70}")

    # 打印相机参数
    print(f"\nCamera A (index 0):")
    print(f"  Image size: {W} x {H}")
    print(f"  FOV: {math.degrees(cam_a.fovx):.2f}° x {math.degrees(cam_a.fovy):.2f}°")

    fx = W / (2 * math.tan(cam_a.fovx / 2.))
    fy = H / (2 * math.tan(cam_a.fovy / 2.))
    print(f"  Focal: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal point: ({W/2}, {H/2})")

    w2c_a = cam_a.world_view_transform.T.cpu().numpy()
    c2w_a = np.linalg.inv(w2c_a)
    print(f"  Camera position: {c2w_a[:3, 3]}")

    print(f"\nCamera B (index 1):")
    w2c_b = cam_b.world_view_transform.T.cpu().numpy()
    c2w_b = np.linalg.inv(w2c_b)
    print(f"  Camera position: {c2w_b[:3, 3]}")

    # 计算相机间的距离
    cam_dist = np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
    print(f"\nDistance between cameras: {cam_dist:.4f}")

    # 渲染两个视角
    print(f"\n{'='*70}")
    print("Rendering both views...")
    print(f"{'='*70}")

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
    alpha_a = render_a['alpha'].cpu().numpy()
    alpha_b = render_b['alpha'].cpu().numpy()

    print(f"Camera A depth range: [{depth_a.min():.4f}, {depth_a.max():.4f}]")
    print(f"Camera B depth range: [{depth_b.min():.4f}, {depth_b.max():.4f}]")

    # 选择一个特定像素进行详细分析
    test_u, test_v = W // 2, H // 2  # 中心像素
    d_a = depth_a[test_v, test_u]

    print(f"\n{'='*70}")
    print(f"Detailed analysis for center pixel ({test_u}, {test_v})")
    print(f"{'='*70}")
    print(f"Depth from Camera A: {d_a:.6f}")

    # 投影到 3D
    cx, cy = W / 2., H / 2.

    # 方法 1: 使用 z-depth 公式
    x_cam = (test_u - cx) / fx * d_a
    y_cam = (test_v - cy) / fy * d_a
    z_cam = d_a
    point_cam_a = np.array([x_cam, y_cam, z_cam, 1.0])

    # 转换到世界坐标
    point_world = c2w_a @ point_cam_a
    print(f"\n3D point (world): ({point_world[0]:.6f}, {point_world[1]:.6f}, {point_world[2]:.6f})")

    # 投影到 Camera B
    point_cam_b = w2c_b @ point_world
    print(f"Point in Camera B space: ({point_cam_b[0]:.6f}, {point_cam_b[1]:.6f}, {point_cam_b[2]:.6f})")

    if point_cam_b[2] > 0:
        u_b = point_cam_b[0] / point_cam_b[2] * fx + cx
        v_b = point_cam_b[1] / point_cam_b[2] * fy + cy
        z_b = point_cam_b[2]
        print(f"Projected pixel in B: ({u_b:.2f}, {v_b:.2f})")
        print(f"Projected depth (z): {z_b:.6f}")

        if 0 <= int(u_b) < W and 0 <= int(v_b) < H:
            d_b_rendered = depth_b[int(v_b), int(u_b)]
            print(f"Rendered depth in B: {d_b_rendered:.6f}")
            print(f"Difference: {abs(z_b - d_b_rendered):.6f}")

            # 检查 alpha
            alpha_at_pixel = alpha_b[int(v_b), int(u_b)]
            print(f"Alpha at this pixel: {alpha_at_pixel:.4f}")

    # 分析可能的问题
    print(f"\n{'='*70}")
    print("Potential issues analysis")
    print(f"{'='*70}")

    # 检查 scene_scaling
    print(f"\nscene_scaling: {model.scene_scaling}")

    # 如果深度没有乘以 scene_scaling，但相机位置已经被缩放了...
    # 或者反过来

    # 查看模型的 center 参数
    print(f"model.center: {model.center}")

    # 检查 transform
    print(f"\nscene_info.transform:\n{scene_info.transform}")

    # 计算从 Camera A 到目标点的射线长度
    ray_dir = point_world[:3] - c2w_a[:3, 3]
    ray_len = np.linalg.norm(ray_dir)
    print(f"\nRay length from Camera A to point: {ray_len:.6f}")
    print(f"z-depth from Camera A: {d_a:.6f}")
    print(f"Ratio (ray_len / z-depth): {ray_len / d_a:.6f}")

    # 如果深度是 t-depth，这个比值应该是 1
    # 如果深度是 z-depth，这个比值应该等于 1/cos(theta)

    # 计算 cos(theta)
    cam_z_dir = c2w_a[:3, :3] @ np.array([0, 0, 1])  # 相机 z 轴方向
    ray_dir_norm = ray_dir / ray_len
    cos_theta = np.dot(ray_dir_norm, cam_z_dir)
    print(f"cos(theta): {cos_theta:.6f}")
    print(f"Expected ratio if z-depth: {1/cos_theta:.6f}")

    # 如果 ratio ≈ 1，深度是 t-depth
    # 如果 ratio ≈ 1/cos(theta)，深度是 z-depth

    # 测试非中心像素
    print(f"\n{'='*70}")
    print("Testing corner pixel")
    print(f"{'='*70}")

    test_u2, test_v2 = W // 4, H // 4
    d_a2 = depth_a[test_v2, test_u2]
    print(f"Corner pixel ({test_u2}, {test_v2}) depth: {d_a2:.6f}")

    # 投影到 3D
    x_cam2 = (test_u2 - cx) / fx * d_a2
    y_cam2 = (test_v2 - cy) / fy * d_a2
    z_cam2 = d_a2
    point_cam_a2 = np.array([x_cam2, y_cam2, z_cam2, 1.0])

    point_world2 = c2w_a @ point_cam_a2

    ray_dir2 = point_world2[:3] - c2w_a[:3, 3]
    ray_len2 = np.linalg.norm(ray_dir2)
    print(f"Ray length from Camera A to point: {ray_len2:.6f}")
    print(f"z-depth from Camera A: {d_a2:.6f}")
    print(f"Ratio (ray_len / z-depth): {ray_len2 / d_a2:.6f}")

    ray_dir_norm2 = ray_dir2 / ray_len2
    cos_theta2 = np.dot(ray_dir_norm2, cam_z_dir)
    print(f"cos(theta): {cos_theta2:.6f}")
    print(f"Expected ratio if z-depth: {1/cos_theta2:.6f}")

    # 结论
    print(f"\n{'='*70}")
    print("Conclusion")
    print(f"{'='*70}")
    print(f"Center pixel ratio: {ray_len / d_a:.6f} (expected 1.0 if t-depth, {1/cos_theta:.6f} if z-depth)")
    print(f"Corner pixel ratio: {ray_len2 / d_a2:.6f} (expected 1.0 if t-depth, {1/cos_theta2:.6f} if z-depth)")

    if abs(ray_len / d_a - 1.0) < 0.01 and abs(ray_len2 / d_a2 - 1.0) < 0.01:
        print("\n** DEPTH IS T-DEPTH (ray distance), NOT Z-DEPTH! **")
        print("This needs conversion before using with Open3D TSDF!")
    elif abs(ray_len / d_a - 1/cos_theta) < 0.01 and abs(ray_len2 / d_a2 - 1/cos_theta2) < 0.01:
        print("\n** DEPTH IS Z-DEPTH (correct for Open3D) **")
        print("The inconsistency might be due to occlusion or other issues.")
    else:
        print("\n** INCONCLUSIVE - depth type unclear **")


if __name__ == "__main__":
    main()
