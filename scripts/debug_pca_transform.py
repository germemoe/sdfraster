#!/usr/bin/env python3
"""
检查点云和相机的 PCA 变换是否一致
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from data import loader
from utils.args import Args

model_path = Path("/storage/jk2/project/output/lambda_ablation/scan24/es0_ndm0_md0_mn0")
config = Args.load_from_json(str(model_path / "config.json"))

print("=" * 60)
print("PCA 变换分析")
print("=" * 60)

# 1. 加载数据（应用 PCA 变换）
print("\n1. 加载数据集（应用 PCA 变换）...")
train_cameras, test_cameras, scene_info = loader.load_dataset(
    config.dataset_path, config.image_folder,
    data_device='cpu', eval=True, resolution=config.resolution
)

# 2. 变换后的点云
pcd_points = scene_info.point_cloud.points
print(f"\n2. 变换后的点云 (scene_info.point_cloud.points):")
print(f"   Shape: {pcd_points.shape}")
print(f"   X: [{pcd_points[:, 0].min():.4f}, {pcd_points[:, 0].max():.4f}]")
print(f"   Y: [{pcd_points[:, 1].min():.4f}, {pcd_points[:, 1].max():.4f}]")
print(f"   Z: [{pcd_points[:, 2].min():.4f}, {pcd_points[:, 2].max():.4f}]")

# 3. 变换后的相机
all_cam_centers = np.array([c.camera_center.numpy() for c in train_cameras])
print(f"\n3. 变换后的相机中心:")
print(f"   X: [{all_cam_centers[:, 0].min():.4f}, {all_cam_centers[:, 0].max():.4f}]")
print(f"   Y: [{all_cam_centers[:, 1].min():.4f}, {all_cam_centers[:, 1].max():.4f}]")
print(f"   Z: [{all_cam_centers[:, 2].min():.4f}, {all_cam_centers[:, 2].max():.4f}]")

# 4. PCA 变换矩阵
print(f"\n4. PCA 变换矩阵:")
print(f"   scene_info.transform:\n{scene_info.transform}")

# 5. 计算 center 和 scaling（与 init_from_pcd 一致）
ccenters = torch.stack([c.camera_center.reshape(3) for c in train_cameras], dim=0)
center = ccenters.mean(dim=0)
scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()

print(f"\n5. 模型初始化参数（应该与 ckpt 一致）:")
print(f"   center (from cameras): {center.numpy()}")
print(f"   scene_scaling (from cameras): {scaling.item():.4f}")

# 6. 加载 checkpoint 中的值
print(f"\n6. Checkpoint 中的值:")
ckpt = torch.load(model_path / "ckpt.pth")
print(f"   center (from ckpt): {ckpt['center'].cpu().numpy()}")
print(f"   scene_scaling (from ckpt): {ckpt['scene_scaling'].cpu().item():.4f}")

# 7. 对比点云和 contracted_vertices
print(f"\n7. 对比点云和 contracted_vertices:")
contracted = ckpt['contracted_vertices'].cpu().numpy()
print(f"   点云 Z 范围: [{pcd_points[:, 2].min():.4f}, {pcd_points[:, 2].max():.4f}]")
print(f"   contracted_vertices Z 范围: [{contracted[:, 2].min():.4f}, {contracted[:, 2].max():.4f}]")

# 8. 检查是否 contracted_vertices 是未变换的点云
print(f"\n8. 检查 contracted_vertices 是否是未变换的点云...")
# 反向应用 PCA 变换
inv_transform = np.linalg.inv(scene_info.transform)
pcd_hom = np.hstack([pcd_points, np.ones((pcd_points.shape[0], 1))])
pcd_original = (inv_transform @ pcd_hom.T).T[:, :3]
print(f"   反变换后的点云 Z 范围: [{pcd_original[:, 2].min():.4f}, {pcd_original[:, 2].max():.4f}]")

# 9. 检查 contracted_vertices 是否匹配原始点云
# 找 contracted_vertices 的中心
contracted_center = contracted.mean(axis=0)
pcd_original_center = pcd_original.mean(axis=0)
pcd_transformed_center = pcd_points.mean(axis=0)

print(f"\n9. 点云中心对比:")
print(f"   变换后点云中心: {pcd_transformed_center}")
print(f"   原始点云中心: {pcd_original_center}")
print(f"   contracted_vertices 中心: {contracted_center}")
