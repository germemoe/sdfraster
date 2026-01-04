#!/usr/bin/env python3
"""
验证 min_t 偏移是否是深度差异的原因
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
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
print("验证 min_t 偏移假设")
print("=" * 60)

cam = train_cameras[0]
min_t = 0.4  # 渲染时使用的 min_t

print(f"\nmin_t = {min_t}")

# 渲染深度
with torch.no_grad():
    render_pkg = render(
        cam, model,
        scene_scaling=model.scene_scaling,
        min_t=min_t,
        tile_size=config.tile_size
    )
    vol_depth = render_pkg['depth'].cpu().numpy()
    alpha = render_pkg['alpha'].cpu().numpy()

# 几何深度（从之前的脚本）
# 这里简单用一个固定值近似
geo_depth_center = 2.25  # 从之前的输出

# 中心像素分析
H, W = cam.image_height, cam.image_width
cy, cx = H // 2, W // 2

vol_d = vol_depth[cy, cx]
alpha_center = alpha[cy, cx]

print(f"\n中心像素 ({cy}, {cx}):")
print(f"  体渲染深度 (从 ray.o): {vol_d:.4f}")
print(f"  alpha: {alpha_center:.4f}")

# 假设：真实深度 = 体渲染深度 + min_t
# 但这不完全正确，因为 min_t 需要转换到 z-space
# z = t * rd_norm_inv，所以 min_t 也需要乘以 rd_norm_inv

import math
fx = W / (2 * math.tan(cam.fovx / 2.))
fy = H / (2 * math.tan(cam.fovy / 2.))
cx_cam, cy_cam = W / 2., H / 2.

# 中心像素的 rd_norm_inv（在中心处约等于 1）
p_x = (cx - cx_cam) / fx
p_y = (cy - cy_cam) / fy
rd_len = np.sqrt(p_x**2 + p_y**2 + 1.0)
rd_norm_inv = 1.0 / rd_len

print(f"  rd_norm_inv: {rd_norm_inv:.4f}")

# min_t 在 z-space 中的偏移
min_t_z = min_t * rd_norm_inv
print(f"  min_t (t-space): {min_t:.4f}")
print(f"  min_t (z-space): {min_t_z:.4f}")

# 修正后的深度
corrected_depth = vol_d + min_t_z
print(f"\n  修正后深度: {vol_d:.4f} + {min_t_z:.4f} = {corrected_depth:.4f}")
print(f"  几何深度 (参考): {geo_depth_center:.4f}")
print(f"  修正后误差: {abs(corrected_depth - geo_depth_center):.4f}")

# 检查全图
valid_mask = vol_depth > 0
vol_valid = vol_depth[valid_mask]

# 假设几何深度在 [1.68, 2.57] 范围，平均 2.02
geo_mean = 2.02

print(f"\n全图分析:")
print(f"  体渲染深度平均 (从 ray.o): {vol_valid.mean():.4f}")
print(f"  修正后深度平均: {vol_valid.mean() + min_t:.4f}")
print(f"  几何深度平均 (参考): {geo_mean:.4f}")

# 修正后的比例
corrected_ratio = (vol_valid.mean() + min_t) / geo_mean
print(f"\n  原始比例: {vol_valid.mean() / geo_mean:.4f}")
print(f"  修正后比例: {corrected_ratio:.4f}")
