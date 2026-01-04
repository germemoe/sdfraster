#!/usr/bin/env python3
"""
可视化多个相机的深度图，检查一致性
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math
from argparse import ArgumentParser
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render


def main():
    parser = ArgumentParser(description="Visualize depth maps from multiple cameras")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_cams", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "debug_depth_viz"
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

    # 选择相机
    num_cams = min(args.num_cams, len(train_cameras))
    cam_indices = np.linspace(0, len(train_cameras)-1, num_cams, dtype=int)

    fig, axes = plt.subplots(num_cams, 4, figsize=(20, 5*num_cams))

    for i, cam_idx in enumerate(cam_indices):
        cam = train_cameras[cam_idx]
        print(f"\nRendering camera {cam_idx} ({i+1}/{num_cams})...")

        with torch.no_grad():
            render_pkg = render(
                cam, model,
                scene_scaling=model.scene_scaling,
                min_t=getattr(config, 'min_t', 0.4),
                tile_size=config.tile_size
            )

        rgb = render_pkg['render'].cpu().permute(1, 2, 0).numpy()
        depth_exp = render_pkg['depth'].cpu().numpy()
        depth_med = render_pkg['depth_med'].cpu().numpy()
        alpha = render_pkg['alpha'].cpu().numpy()

        # 计算深度统计
        valid_mask = alpha > 0.5
        if valid_mask.any():
            d_exp_valid = depth_exp[valid_mask]
            d_med_valid = depth_med[valid_mask]
            print(f"  Depth expected: min={d_exp_valid.min():.4f}, max={d_exp_valid.max():.4f}, mean={d_exp_valid.mean():.4f}")
            print(f"  Depth median:   min={d_med_valid.min():.4f}, max={d_med_valid.max():.4f}, mean={d_med_valid.mean():.4f}")

        row = axes[i] if num_cams > 1 else axes

        # RGB
        row[0].imshow(np.clip(rgb, 0, 1))
        row[0].set_title(f'Camera {cam_idx}: RGB')
        row[0].axis('off')

        # Depth expected
        im1 = row[1].imshow(depth_exp, cmap='viridis')
        row[1].set_title(f'Depth Expected')
        row[1].axis('off')
        plt.colorbar(im1, ax=row[1], fraction=0.046)

        # Depth median
        im2 = row[2].imshow(depth_med, cmap='viridis')
        row[2].set_title(f'Depth Median')
        row[2].axis('off')
        plt.colorbar(im2, ax=row[2], fraction=0.046)

        # Alpha
        im3 = row[3].imshow(alpha, cmap='gray')
        row[3].set_title(f'Alpha')
        row[3].axis('off')
        plt.colorbar(im3, ax=row[3], fraction=0.046)

        # 保存单独的深度图（用于进一步分析）
        np.save(output_dir / f"depth_exp_cam{cam_idx}.npy", depth_exp)
        np.save(output_dir / f"depth_med_cam{cam_idx}.npy", depth_med)

    plt.tight_layout()
    output_file = output_dir / "depth_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")
    plt.close()

    # 额外检查：计算深度一致性
    print("\n" + "="*60)
    print("Depth Statistics Summary:")
    print("="*60)

    all_depths = []
    for cam_idx in cam_indices:
        depth = np.load(output_dir / f"depth_med_cam{cam_idx}.npy")
        all_depths.append(depth)

    # 打印全局统计
    all_valid = []
    for i, cam_idx in enumerate(cam_indices):
        cam = train_cameras[cam_idx]
        with torch.no_grad():
            render_pkg = render(
                cam, model,
                scene_scaling=model.scene_scaling,
                min_t=getattr(config, 'min_t', 0.4),
                tile_size=config.tile_size
            )
        alpha = render_pkg['alpha'].cpu().numpy()
        depth = all_depths[i]
        valid = depth[alpha > 0.5]
        all_valid.extend(valid.flatten())

    all_valid = np.array(all_valid)
    print(f"All valid depths: min={all_valid.min():.4f}, max={all_valid.max():.4f}")
    print(f"                  mean={all_valid.mean():.4f}, std={all_valid.std():.4f}")
    print(f"                  median={np.median(all_valid):.4f}")

    # 检查是否有异常值
    q1, q3 = np.percentile(all_valid, [25, 75])
    iqr = q3 - q1
    outlier_low = q1 - 1.5 * iqr
    outlier_high = q3 + 1.5 * iqr
    outliers = np.sum((all_valid < outlier_low) | (all_valid > outlier_high))
    print(f"Outlier ratio: {outliers / len(all_valid) * 100:.2f}%")


if __name__ == "__main__":
    main()
