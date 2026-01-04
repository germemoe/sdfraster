#!/usr/bin/env python3
"""
验证深度修复是否正确
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import math
import open3d as o3d
from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render

model_path = Path("/storage/jk2/project/output/lambda_ablation/scan24/es0_ndm0_md0_mn0")
config = Args.load_from_json(str(model_path / "config.json"))

# 加载数据
train_cameras, _, scene_info = loader.load_dataset(
    config.dataset_path, config.image_folder,
    data_device='cpu', eval=True, resolution=config.resolution
)

# 加载模型
device = torch.device('cuda')
model = Model.load_ckpt(model_path, device)
model.eval()

print("=" * 60)
print("验证深度修复")
print("=" * 60)

cam = train_cameras[0]
H, W = cam.image_height, cam.image_width
min_t = 0.4

# 渲染深度
with torch.no_grad():
    render_pkg = render(
        cam, model,
        scene_scaling=model.scene_scaling,
        min_t=min_t,
        tile_size=config.tile_size
    )
    expected_depth = render_pkg['depth']  # (H, W)
    median_depth = render_pkg['depth_med']  # (H, W)

depth_ratio = 0.0  # MILo DTU default
surface_depth = depth_ratio * expected_depth + (1.0 - depth_ratio) * median_depth

# 计算 min_t 偏移
fx = W / (2 * math.tan(cam.fovx / 2.))
fy = H / (2 * math.tan(cam.fovy / 2.))
cx, cy = W / 2., H / 2.

y_coords, x_coords = torch.meshgrid(
    torch.arange(H, dtype=torch.float32, device=device) + 0.5,
    torch.arange(W, dtype=torch.float32, device=device) + 0.5,
    indexing='ij'
)
p_x = (x_coords - cx) / fx
p_y = (y_coords - cy) / fy
rd_norm_inv = 1.0 / torch.sqrt(1.0 + p_x**2 + p_y**2)

min_t_z = min_t * rd_norm_inv
valid_mask = surface_depth > 0
corrected_depth = surface_depth + min_t_z.to(surface_depth.device) * valid_mask.float()

# 用 MT mesh 渲染几何深度作为参考
mesh_file = model_path / "mesh_mt.ply"
mesh = o3d.io.read_triangle_mesh(str(mesh_file))
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_t)

# 生成射线
y_np, x_np = np.meshgrid(
    np.arange(H, dtype=np.float32) + 0.5,
    np.arange(W, dtype=np.float32) + 0.5,
    indexing='ij'
)
dirs_cam = np.stack([
    (x_np - cx) / fx,
    (y_np - cy) / fy,
    np.ones_like(x_np)
], axis=-1)

w2c = cam.world_view_transform.T.cpu().numpy()
c2w = np.linalg.inv(w2c)
R_c2w = c2w[:3, :3]
cam_pos = c2w[:3, 3]

dirs_world = dirs_cam @ R_c2w.T
dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
origins = np.broadcast_to(cam_pos, dirs_world.shape)
rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32).reshape(-1, 6)

result = scene.cast_rays(o3d.core.Tensor(rays))
t_hit = result['t_hit'].numpy().reshape(H, W)

d_unnorm_len = np.sqrt(((x_np - cx) / fx)**2 + ((y_np - cy) / fy)**2 + 1.0)
geo_depth = t_hit / d_unnorm_len
geo_depth[t_hit == np.inf] = 0

# 对比
print(f"\n中心像素 ({H//2}, {W//2}):")
cy_int, cx_int = H // 2, W // 2
print(f"  原始体渲染深度: {surface_depth[cy_int, cx_int].item():.4f}")
print(f"  修正后深度: {corrected_depth[cy_int, cx_int].item():.4f}")
print(f"  几何深度: {geo_depth[cy_int, cx_int]:.4f}")

if geo_depth[cy_int, cx_int] > 0:
    orig_error = abs(surface_depth[cy_int, cx_int].item() - geo_depth[cy_int, cx_int])
    corrected_error = abs(corrected_depth[cy_int, cx_int].item() - geo_depth[cy_int, cx_int])
    print(f"  原始误差: {orig_error:.4f}")
    print(f"  修正后误差: {corrected_error:.4f}")
    print(f"  误差减少: {orig_error - corrected_error:.4f} ({(1 - corrected_error/orig_error)*100:.1f}%)")

# 全图统计
vol_valid = (surface_depth > 0).cpu().numpy()
geo_valid = geo_depth > 0
both_valid = vol_valid & geo_valid

if both_valid.sum() > 0:
    orig_d = surface_depth.cpu().numpy()[both_valid]
    corr_d = corrected_depth.cpu().numpy()[both_valid]
    geo_d = geo_depth[both_valid]

    print(f"\n全图统计 ({both_valid.sum()} 有效像素):")
    print(f"  原始比例 (vol/geo): {(orig_d / geo_d).mean():.4f}")
    print(f"  修正后比例 (corr/geo): {(corr_d / geo_d).mean():.4f}")
    print(f"  原始差值: mean={np.abs(orig_d - geo_d).mean():.4f}")
    print(f"  修正后差值: mean={np.abs(corr_d - geo_d).mean():.4f}")
