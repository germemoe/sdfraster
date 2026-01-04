#!/usr/bin/env python3
"""
调试坐标系一致性：相机 vs 模型顶点 vs 深度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import math
from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render
from utils.tetmesh import marching_tetrahedra

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
print("坐标系分析")
print("=" * 60)

# 1. 相机中心
cam = train_cameras[0]
cam_center = cam.camera_center.numpy()
print(f"\n1. 相机中心 (camera_center):")
print(f"   Camera 0: {cam_center}")

# 所有相机
all_cam_centers = np.array([c.camera_center.numpy() for c in train_cameras])
print(f"   All cameras X: [{all_cam_centers[:, 0].min():.4f}, {all_cam_centers[:, 0].max():.4f}]")
print(f"   All cameras Y: [{all_cam_centers[:, 1].min():.4f}, {all_cam_centers[:, 1].max():.4f}]")
print(f"   All cameras Z: [{all_cam_centers[:, 2].min():.4f}, {all_cam_centers[:, 2].max():.4f}]")

# 2. 模型的 center 和 scene_scaling
print(f"\n2. 模型参数:")
print(f"   model.center: {model.center.cpu().numpy()}")
print(f"   model.scene_scaling: {model.scene_scaling.item():.4f}")

# 3. 模型顶点
verts = model.vertices.detach().cpu().numpy()
print(f"\n3. 模型顶点 (model.vertices):")
print(f"   Shape: {verts.shape}")
print(f"   X: [{verts[:, 0].min():.4f}, {verts[:, 0].max():.4f}]")
print(f"   Y: [{verts[:, 1].min():.4f}, {verts[:, 1].max():.4f}]")
print(f"   Z: [{verts[:, 2].min():.4f}, {verts[:, 2].max():.4f}]")

# 4. 内部顶点（contracted_vertices）
int_verts = model.contracted_vertices.detach().cpu().numpy()
print(f"\n4. 内部顶点 (contracted_vertices):")
print(f"   Shape: {int_verts.shape}")
print(f"   X: [{int_verts[:, 0].min():.4f}, {int_verts[:, 0].max():.4f}]")
print(f"   Y: [{int_verts[:, 1].min():.4f}, {int_verts[:, 1].max():.4f}]")
print(f"   Z: [{int_verts[:, 2].min():.4f}, {int_verts[:, 2].max():.4f}]")

# 5. 外部顶点
ext_verts = model.ext_vertices.detach().cpu().numpy()
print(f"\n5. 外部顶点 (ext_vertices):")
print(f"   Shape: {ext_verts.shape}")
print(f"   X: [{ext_verts[:, 0].min():.4f}, {ext_verts[:, 0].max():.4f}]")
print(f"   Y: [{ext_verts[:, 1].min():.4f}, {ext_verts[:, 1].max():.4f}]")
print(f"   Z: [{ext_verts[:, 2].min():.4f}, {ext_verts[:, 2].max():.4f}]")

# 6. MT mesh 的顶点
print(f"\n6. MT Mesh 提取:")
with torch.no_grad():
    vertices = model.vertices
    indices = model.indices
    vert_sdf = model.query_vert_sdf()

    # marching_tetrahedra 返回的顶点
    mt_verts, mt_faces = marching_tetrahedra(vertices, indices, vert_sdf, 0.0)
    mt_verts_np = mt_verts.cpu().numpy()

print(f"   MT vertices shape: {mt_verts_np.shape}")
print(f"   X: [{mt_verts_np[:, 0].min():.4f}, {mt_verts_np[:, 0].max():.4f}]")
print(f"   Y: [{mt_verts_np[:, 1].min():.4f}, {mt_verts_np[:, 1].max():.4f}]")
print(f"   Z: [{mt_verts_np[:, 2].min():.4f}, {mt_verts_np[:, 2].max():.4f}]")

# 7. 渲染一个视角的深度
print(f"\n7. 渲染深度分析:")
cam = train_cameras[0]
with torch.no_grad():
    render_pkg = render(
        cam, model,
        scene_scaling=model.scene_scaling,
        min_t=0.4,
        tile_size=config.tile_size
    )
    depth = render_pkg['depth'].cpu().numpy()
    depth_med = render_pkg['depth_med'].cpu().numpy()

valid_mask = depth > 0
print(f"   Expected depth: min={depth[valid_mask].min():.4f}, max={depth[valid_mask].max():.4f}, mean={depth[valid_mask].mean():.4f}")
print(f"   Median depth: min={depth_med[valid_mask].min():.4f}, max={depth_med[valid_mask].max():.4f}, mean={depth_med[valid_mask].mean():.4f}")

# 8. 理论深度计算
print(f"\n8. 理论深度 (相机中心到 MT mesh 的距离):")
cam_pos = cam.camera_center.numpy()
# 计算相机到 MT mesh 各点的 z-depth
# 需要用相机的 world_view_transform
w2c = cam.world_view_transform.T.numpy()  # W2C 矩阵

# 将 MT mesh 顶点变换到相机坐标系
mt_verts_hom = np.hstack([mt_verts_np, np.ones((mt_verts_np.shape[0], 1))])
mt_verts_cam = (w2c @ mt_verts_hom.T).T[:, :3]

print(f"   MT mesh in camera space:")
print(f"   Z (depth): [{mt_verts_cam[:, 2].min():.4f}, {mt_verts_cam[:, 2].max():.4f}]")

# 9. 对比分析
print(f"\n9. 深度比例分析:")
vol_depth_mean = depth[valid_mask].mean()
mt_z_mean = mt_verts_cam[:, 2].mean()
print(f"   体渲染平均深度: {vol_depth_mean:.4f}")
print(f"   MT mesh 平均 Z (相机空间): {mt_z_mean:.4f}")
print(f"   比例 (体渲染/MT): {vol_depth_mean / mt_z_mean:.4f}")
print(f"   scene_scaling: {model.scene_scaling.item():.4f}")
print(f"   1/scene_scaling: {1/model.scene_scaling.item():.4f}")

# 10. 检查射线原点和方向
print(f"\n10. 射线参数 (来自 camera.to_dict):")
cam_dict = cam.to_dict(device)
print(f"   cam_pos: {cam_dict['cam_pos'].cpu().numpy()}")
print(f"   K (内参):\n{cam_dict['K'].cpu().numpy()}")

# world_view_transform 的格式
wvt = cam_dict['world_view_transform'].cpu().numpy()
print(f"   world_view_transform (W2C):\n{wvt}")

# 验证 camera_center = inv(W2C)[3, :3]
w2c_inv = np.linalg.inv(wvt)
print(f"   inv(W2C)[3, :3] = {w2c_inv[3, :3]} (应该等于 cam_pos)")

# 11. 检查相机中心是否在 model.center 附近
print(f"\n11. 相机中心相对于 model.center 的位置:")
center = model.center.cpu().numpy().flatten()
rel_cam = cam_pos - center
print(f"   相机中心相对位置: {rel_cam}")
print(f"   相对距离: {np.linalg.norm(rel_cam):.4f}")
print(f"   相对距离 / scene_scaling: {np.linalg.norm(rel_cam) / model.scene_scaling.item():.4f}")
