#!/usr/bin/env python3
"""
检查深度差异是否是常数倍关系
"""

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

    train_cameras, _, scene_info = loader.load_dataset(
        config.dataset_path, config.image_folder,
        data_device='cpu', eval=True, resolution=config.resolution
    )

    device = torch.device('cuda')
    model = Model.load_ckpt(model_path, device)
    model.eval()

    cam_a = train_cameras[0]
    cam_b = train_cameras[1]

    H = cam_a.image_height
    W = cam_a.image_width
    fx = W / (2 * math.tan(cam_a.fovx / 2.))
    fy = H / (2 * math.tan(cam_a.fovy / 2.))
    cx, cy = W / 2., H / 2.

    print(f"scene_scaling = {model.scene_scaling}")

    with torch.no_grad():
        render_a = render(cam_a, model, scene_scaling=model.scene_scaling,
                         min_t=getattr(config, 'min_t', 0.4), tile_size=config.tile_size)
        render_b = render(cam_b, model, scene_scaling=model.scene_scaling,
                         min_t=getattr(config, 'min_t', 0.4), tile_size=config.tile_size)

    depth_a = render_a['depth_med'].cpu().numpy()
    depth_b = render_b['depth_med'].cpu().numpy()

    w2c_a = cam_a.world_view_transform.T.cpu().numpy()
    c2w_a = np.linalg.inv(w2c_a)
    w2c_b = cam_b.world_view_transform.T.cpu().numpy()

    # 采样一些像素点
    test_pixels = [
        (W // 2, H // 2),
        (W // 4, H // 4),
        (3 * W // 4, H // 4),
        (W // 4, 3 * H // 4),
        (3 * W // 4, 3 * H // 4),
    ]

    print(f"\n{'='*80}")
    print("Depth ratio analysis")
    print(f"{'='*80}")
    print(f"{'Pixel':<15} {'D_A':<10} {'D_B_proj':<10} {'D_B_rend':<10} {'Ratio':<10} {'Diff':<10}")
    print("-" * 80)

    ratios = []
    for u, v in test_pixels:
        d_a = depth_a[v, u]
        if d_a <= 0:
            continue

        # 投影到 3D
        x_cam = (u - cx) / fx * d_a
        y_cam = (v - cy) / fy * d_a
        z_cam = d_a
        point_cam_a = np.array([x_cam, y_cam, z_cam, 1.0])
        point_world = c2w_a @ point_cam_a

        # 投影到 Camera B
        point_cam_b = w2c_b @ point_world
        if point_cam_b[2] <= 0:
            continue

        u_b = int(point_cam_b[0] / point_cam_b[2] * fx + cx)
        v_b = int(point_cam_b[1] / point_cam_b[2] * fy + cy)

        if 0 <= u_b < W and 0 <= v_b < H:
            z_proj = point_cam_b[2]
            d_b_rend = depth_b[v_b, u_b]

            if d_b_rend > 0:
                ratio = d_b_rend / z_proj
                diff = d_b_rend - z_proj
                ratios.append(ratio)
                print(f"({u:4d},{v:4d})    {d_a:<10.4f} {z_proj:<10.4f} {d_b_rend:<10.4f} {ratio:<10.4f} {diff:<10.4f}")

    if ratios:
        print("-" * 80)
        print(f"Mean ratio: {np.mean(ratios):.6f}")
        print(f"Std ratio:  {np.std(ratios):.6f}")
        print(f"scene_scaling: {model.scene_scaling}")
        print(f"1/scene_scaling: {1/model.scene_scaling:.6f}")
        print(f"scene_scaling^2: {model.scene_scaling**2:.6f}")


if __name__ == "__main__":
    main()
