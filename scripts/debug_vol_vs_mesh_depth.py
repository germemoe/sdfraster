#!/usr/bin/env python3
"""
比较体渲染深度和几何（mesh）渲染深度的比值
"""

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


def render_mesh_depth(mesh_path, cam, device='cuda'):
    """用 Open3D 渲染 mesh 的深度图"""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    H = cam.image_height
    W = cam.image_width
    fx = W / (2 * math.tan(cam.fovx / 2.))
    fy = H / (2 * math.tan(cam.fovy / 2.))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W/2, H/2)
    extrinsic = cam.world_view_transform.T.cpu().numpy()

    # 创建渲染器
    vis = o3d.visualization.rendering.OffscreenRenderer(W, H)
    vis.scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())
    vis.setup_camera(intrinsic, extrinsic)

    depth = np.asarray(vis.render_to_depth_image(z_in_view_space=True))
    return depth


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

    # 找 MT mesh 文件
    mesh_files = list(model_path.glob("mesh_mt*.ply")) + list(model_path.glob("mesh*.ply"))
    mesh_path = None
    for f in mesh_files:
        if 'tsdf' not in f.name and 'world' not in f.name and 'culled' not in f.name and 'post' not in f.name:
            mesh_path = f
            break

    if mesh_path is None:
        print("No mesh file found!")
        return

    print(f"Using mesh: {mesh_path}")
    print(f"scene_scaling = {model.scene_scaling}")

    cam = train_cameras[0]

    # 体渲染深度
    with torch.no_grad():
        render_pkg = render(cam, model, scene_scaling=model.scene_scaling,
                           min_t=getattr(config, 'min_t', 0.4), tile_size=config.tile_size)

    depth_vol = render_pkg['depth_med'].cpu().numpy()
    alpha = render_pkg['alpha'].cpu().numpy()

    # mesh 渲染深度
    depth_mesh = render_mesh_depth(mesh_path, cam)

    # 只比较有效区域
    valid_mask = (alpha > 0.5) & (depth_mesh > 0) & (depth_mesh < 100) & (depth_vol > 0)

    depth_vol_valid = depth_vol[valid_mask]
    depth_mesh_valid = depth_mesh[valid_mask]

    ratio = depth_vol_valid / depth_mesh_valid

    print(f"\n{'='*60}")
    print("Volume rendering depth vs Mesh depth")
    print(f"{'='*60}")
    print(f"Valid pixels: {valid_mask.sum()}")
    print(f"\nDepth_vol / Depth_mesh ratio:")
    print(f"  Mean:   {ratio.mean():.6f}")
    print(f"  Std:    {ratio.std():.6f}")
    print(f"  Median: {np.median(ratio):.6f}")
    print(f"  Min:    {ratio.min():.6f}")
    print(f"  Max:    {ratio.max():.6f}")

    print(f"\n1/scene_scaling = {1/model.scene_scaling:.6f}")
    print(f"scene_scaling = {model.scene_scaling:.6f}")

    # 检查特定像素
    H, W = depth_vol.shape
    test_pixels = [(W//2, H//2), (W//4, H//4), (3*W//4, H//2)]

    print(f"\n{'='*60}")
    print("Sample pixels")
    print(f"{'='*60}")
    print(f"{'Pixel':<15} {'D_vol':<12} {'D_mesh':<12} {'Ratio':<12}")
    print("-" * 60)
    for u, v in test_pixels:
        if valid_mask[v, u]:
            print(f"({u:4d},{v:4d})    {depth_vol[v,u]:<12.4f} {depth_mesh[v,u]:<12.4f} {depth_vol[v,u]/depth_mesh[v,u]:<12.6f}")


if __name__ == "__main__":
    main()
