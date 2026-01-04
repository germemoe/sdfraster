#!/usr/bin/env python3
"""
诊断 TSDF 深度问题的脚本

检查渲染深度是 t-depth 还是 z-depth，并可视化差异。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render


def compute_rd_norm_inv_map(H, W, K):
    """
    计算整个图像的 rd_norm_inv map

    rd_norm_inv = 1 / sqrt(p_x^2 + p_y^2 + 1)
    其中 p_x = (x - cx) / fx, p_y = (y - cy) / fy

    对于像素 (x, y)，使用像素中心 (x + 0.5, y + 0.5)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 创建像素坐标网格 (使用像素中心)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32) + 0.5,
        torch.arange(W, dtype=torch.float32) + 0.5,
        indexing='ij'
    )

    p_x = (x_coords - cx) / fx
    p_y = (y_coords - cy) / fy

    cam_rd_len_sq = p_x ** 2 + p_y ** 2 + 1.0
    rd_norm_inv = 1.0 / torch.sqrt(cam_rd_len_sq)

    return rd_norm_inv


def t_depth_to_z_depth(t_depth, rd_norm_inv):
    """将 t-depth 转换为 z-depth"""
    return t_depth * rd_norm_inv


def z_depth_to_t_depth(z_depth, rd_norm_inv):
    """将 z-depth 转换为 t-depth"""
    return z_depth / rd_norm_inv


def main():
    parser = ArgumentParser(description="Debug TSDF depth conversion")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cam_idx", type=int, default=0, help="Camera index to analyze")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "debug_tsdf"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 加载配置
    config_path = model_path / "config.json"
    config = Args.load_from_json(str(config_path))

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
    model.eval()

    # 选择一个相机
    cam = train_cameras[args.cam_idx]
    print(f"\nAnalyzing camera {args.cam_idx}:")
    print(f"  Image size: {cam.image_width} x {cam.image_height}")
    print(f"  FoVx: {math.degrees(cam.fovx):.2f} deg")
    print(f"  FoVy: {math.degrees(cam.fovy):.2f} deg")

    # 计算相机内参
    W, H = cam.image_width, cam.image_height
    fx = W / (2 * math.tan(cam.fovx / 2.))
    fy = H / (2 * math.tan(cam.fovy / 2.))
    cx, cy = W / 2., H / 2.

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)

    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # 计算 rd_norm_inv map
    rd_norm_inv_map = compute_rd_norm_inv_map(H, W, K)

    print(f"\nrd_norm_inv statistics:")
    print(f"  Min: {rd_norm_inv_map.min():.6f}")
    print(f"  Max: {rd_norm_inv_map.max():.6f}")
    print(f"  Center: {rd_norm_inv_map[H//2, W//2]:.6f}")
    print(f"  Corner: {rd_norm_inv_map[0, 0]:.6f}")

    # 渲染深度
    print("\nRendering depth...")
    with torch.no_grad():
        render_pkg = render(
            cam, model,
            scene_scaling=model.scene_scaling,
            min_t=getattr(config, 'min_t', 0.4),
            tile_size=config.tile_size
        )

    depth_expected = render_pkg['depth'].cpu()  # (H, W)
    depth_med = render_pkg['depth_med'].cpu()   # (H, W)

    print(f"\nDepth statistics (from slang):")
    print(f"  Expected depth - min: {depth_expected.min():.4f}, max: {depth_expected.max():.4f}")
    print(f"  Median depth - min: {depth_med.min():.4f}, max: {depth_med.max():.4f}")

    # 检查深度是否已经是 z-depth
    # 方法：如果是 t-depth，转换到 z-depth 后值应该更小（尤其在边缘）
    # 如果转换后值几乎不变（ratio ≈ 1），说明 slang 已输出 z-depth

    # 假设当前是 t-depth，转换为 z-depth
    depth_med_as_z = t_depth_to_z_depth(depth_med, rd_norm_inv_map)

    # 计算转换比例
    valid_mask = depth_med > 0
    if valid_mask.any():
        ratio = (depth_med_as_z / depth_med)[valid_mask]
        print(f"\nIf treating slang output as t-depth:")
        print(f"  Conversion ratio min: {ratio.min():.6f}")
        print(f"  Conversion ratio max: {ratio.max():.6f}")
        print(f"  Conversion ratio mean: {ratio.mean():.6f}")

        # 如果 ratio 非常接近 rd_norm_inv 的范围，说明确实是 t-depth
        # 如果 ratio 几乎都是 1，说明 slang 已经输出 z-depth

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: 深度图
    im0 = axes[0, 0].imshow(depth_med.numpy(), cmap='viridis')
    axes[0, 0].set_title('Median Depth (from slang)')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(rd_norm_inv_map.numpy(), cmap='coolwarm')
    axes[0, 1].set_title('rd_norm_inv map')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(depth_med_as_z.numpy(), cmap='viridis')
    axes[0, 2].set_title('Depth after t→z conversion')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: 差异分析
    diff = depth_med - depth_med_as_z
    im3 = axes[1, 0].imshow(diff.numpy(), cmap='RdBu')
    axes[1, 0].set_title('Difference (original - converted)')
    plt.colorbar(im3, ax=axes[1, 0])

    # 相对差异
    rel_diff = diff / depth_med.clamp(min=0.001)
    im4 = axes[1, 1].imshow(rel_diff.numpy() * 100, cmap='RdBu', vmin=-10, vmax=10)
    axes[1, 1].set_title('Relative difference (%)')
    plt.colorbar(im4, ax=axes[1, 1])

    # 显示 GT image
    gt_image = cam.original_image.permute(1, 2, 0).cpu().numpy()
    axes[1, 2].imshow(gt_image)
    axes[1, 2].set_title('GT Image')

    plt.tight_layout()
    output_file = output_dir / f"depth_analysis_cam{args.cam_idx}.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved analysis to: {output_file}")

    # 检查中心线剖面
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)

    # 判断 slang 输出是 t-depth 还是 z-depth
    # 通过检查 rd_norm_inv 的变化范围
    rd_norm_inv_range = rd_norm_inv_map.max() - rd_norm_inv_map.min()

    if rd_norm_inv_range > 0.01:  # 有明显的空间变化
        print(f"rd_norm_inv varies from {rd_norm_inv_map.min():.4f} to {rd_norm_inv_map.max():.4f}")
        print(f"This means t-depth and z-depth can differ by up to {(1 - rd_norm_inv_map.min()) * 100:.1f}% at corners")
        print()
        print("To determine if slang outputs t-depth or z-depth:")
        print("  - If a flat wall appears curved in depth map → likely t-depth")
        print("  - If a flat wall appears flat in depth map → likely z-depth")
        print()
        print("For OLD models (before rd_norm_inv fix):")
        print("  The slang outputs t-depth. You need to apply conversion in Python:")
        print("  z_depth = t_depth * rd_norm_inv")
        print()
        print("For NEW models (after rd_norm_inv fix):")
        print("  The slang outputs z-depth directly. No conversion needed.")

    plt.close()


if __name__ == "__main__":
    main()
