#!/usr/bin/env python3
"""
可视化异常深度相机
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render


def main():
    model_path = Path('/home/jk/jk2/project/output/lambda_ablation/scan24/es0_ndm0_md0_mn0')
    config = Args.load_from_json(str(model_path / "config.json"))

    train_cameras, _, _ = loader.load_dataset(
        config.dataset_path, config.image_folder,
        data_device='cpu', eval=True, resolution=config.resolution
    )

    device = torch.device('cuda')
    model = Model.load_ckpt(model_path, device)
    model.eval()

    # 渲染有问题的相机
    cam_indices = [0, 10, 20, 30, 40]

    fig, axes = plt.subplots(len(cam_indices), 4, figsize=(16, 4*len(cam_indices)))

    for row, cam_idx in enumerate(cam_indices):
        cam = train_cameras[cam_idx]
        with torch.no_grad():
            render_pkg = render(
                cam, model,
                scene_scaling=model.scene_scaling,
                min_t=getattr(config, 'min_t', 0.4),
                tile_size=config.tile_size
            )

        rgb = render_pkg['render'].cpu().permute(1, 2, 0).numpy()
        depth_med = render_pkg['depth_med'].cpu().numpy()
        alpha = render_pkg['alpha'].cpu().numpy()

        valid_mask = alpha > 0.5
        valid_depth = depth_med[valid_mask] if valid_mask.any() else depth_med.flatten()

        # RGB
        axes[row, 0].imshow(np.clip(rgb, 0, 1))
        axes[row, 0].set_title(f'Cam {cam_idx}: RGB')
        axes[row, 0].axis('off')

        # Depth (clipped to reasonable range)
        depth_clipped = np.clip(depth_med, 0, 5)
        im = axes[row, 1].imshow(depth_clipped, cmap='viridis')
        axes[row, 1].set_title(f'Depth (clipped [0,5])')
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046)

        # Depth (full range)
        im2 = axes[row, 2].imshow(depth_med, cmap='viridis')
        axes[row, 2].set_title(f'Depth (full) [{valid_depth.min():.2f}, {valid_depth.max():.2f}]')
        axes[row, 2].axis('off')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        # Alpha
        axes[row, 3].imshow(alpha, cmap='gray')
        axes[row, 3].set_title(f'Alpha')
        axes[row, 3].axis('off')

    plt.tight_layout()
    output_path = model_path / 'debug_depth_viz' / 'depth_outliers.png'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=100)
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
