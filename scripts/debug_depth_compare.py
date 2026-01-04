#!/usr/bin/env python3
"""
精确比较体渲染深度和几何深度
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
print("精确深度比较")
print("=" * 60)

# 选择第一个相机
cam = train_cameras[0]
H, W = cam.image_height, cam.image_width

print(f"\n相机参数:")
print(f"  H x W: {H} x {W}")
print(f"  fovx: {cam.fovx:.4f}, fovy: {cam.fovy:.4f}")
print(f"  camera_center: {cam.camera_center.numpy()}")

# 1. 体渲染深度
print(f"\n1. 体渲染深度...")
with torch.no_grad():
    render_pkg = render(
        cam, model,
        scene_scaling=model.scene_scaling,
        min_t=0.4,
        tile_size=config.tile_size
    )
    vol_depth = render_pkg['depth'].cpu().numpy()
    vol_depth_med = render_pkg['depth_med'].cpu().numpy()

valid_mask = vol_depth > 0
print(f"   Expected depth: min={vol_depth[valid_mask].min():.4f}, max={vol_depth[valid_mask].max():.4f}, mean={vol_depth[valid_mask].mean():.4f}")
print(f"   Median depth: min={vol_depth_med[valid_mask].min():.4f}, max={vol_depth_med[valid_mask].max():.4f}, mean={vol_depth_med[valid_mask].mean():.4f}")

# 2. 用 MT mesh 渲染深度（真实几何）
print(f"\n2. MT Mesh 几何深度...")

# 提取 MT mesh
with torch.no_grad():
    vertices = model.vertices
    indices = model.indices
    vert_sdf = model.query_vert_sdf()
    mt_verts, mt_faces = marching_tetrahedra(vertices, indices, vert_sdf, 0.0)

if mt_verts.shape[0] == 0:
    print("   警告: MT mesh 为空！检查 SDF...")
    # 检查 SDF 分布
    sdf_vals = vert_sdf.cpu().numpy()
    print(f"   SDF 范围: [{sdf_vals.min():.4f}, {sdf_vals.max():.4f}]")
    print(f"   SDF > 0: {(sdf_vals > 0).sum()}, SDF < 0: {(sdf_vals < 0).sum()}")

    # 尝试直接从已保存的 mesh 加载
    mesh_files = list(model_path.glob("mesh_mt*.ply"))
    if mesh_files:
        print(f"\n   从文件加载 MT mesh: {mesh_files[0]}")
        mesh = o3d.io.read_triangle_mesh(str(mesh_files[0]))
        mt_verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
        mt_faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int32)
    else:
        print("   没有找到 MT mesh 文件！")
        exit(1)

print(f"   MT mesh: {mt_verts.shape[0]} vertices, {mt_faces.shape[0]} faces")

# 用 Open3D raycasting 渲染深度
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(mt_verts.cpu().numpy())
mesh.triangles = o3d.utility.Vector3iVector(mt_faces.cpu().numpy())

mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_t)

# 生成射线
fx = W / (2 * math.tan(cam.fovx / 2.))
fy = H / (2 * math.tan(cam.fovy / 2.))
cx, cy = W / 2., H / 2.

y_coords, x_coords = np.meshgrid(
    np.arange(H, dtype=np.float32) + 0.5,
    np.arange(W, dtype=np.float32) + 0.5,
    indexing='ij'
)

# 相机坐标系方向
dirs_cam = np.stack([
    (x_coords - cx) / fx,
    (y_coords - cy) / fy,
    np.ones_like(x_coords)
], axis=-1)

# 获取 C2W 变换
w2c = cam.world_view_transform.T.cpu().numpy()  # W2C
c2w = np.linalg.inv(w2c)

R_c2w = c2w[:3, :3]
cam_pos = c2w[:3, 3]

# 世界坐标系方向
dirs_world = dirs_cam @ R_c2w.T
dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)

# 射线原点
origins = np.broadcast_to(cam_pos, dirs_world.shape)
rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32).reshape(-1, 6)

# 光线追踪
result = scene.cast_rays(o3d.core.Tensor(rays))
t_hit = result['t_hit'].numpy().reshape(H, W)

# 转换为 z-depth
d_unnorm_len = np.sqrt(((x_coords - cx) / fx)**2 + ((y_coords - cy) / fy)**2 + 1.0)
geo_depth = t_hit / d_unnorm_len
geo_depth[t_hit == np.inf] = 0

geo_valid = geo_depth > 0
print(f"   Geometric depth: min={geo_depth[geo_valid].min():.4f}, max={geo_depth[geo_valid].max():.4f}, mean={geo_depth[geo_valid].mean():.4f}")

# 3. 比较
print(f"\n3. 深度比较 (有效像素):")
both_valid = valid_mask & geo_valid
if both_valid.sum() > 0:
    vol_d = vol_depth[both_valid]
    geo_d = geo_depth[both_valid]

    diff = vol_d - geo_d
    ratio = vol_d / geo_d

    print(f"   有效像素数: {both_valid.sum()}")
    print(f"   差值 (vol - geo): mean={diff.mean():.4f}, std={diff.std():.4f}")
    print(f"   比例 (vol / geo): mean={ratio.mean():.4f}, std={ratio.std():.4f}")

    # 按百分位分析
    print(f"\n   比例分布:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"     {p}%: {np.percentile(ratio, p):.4f}")

    # 检查是否是常数倍
    print(f"\n   如果是常数倍差异:")
    print(f"     vol_depth ≈ {ratio.mean():.4f} * geo_depth")
    print(f"     scene_scaling = {model.scene_scaling.item():.4f}")
    print(f"     1/scene_scaling = {1/model.scene_scaling.item():.4f}")

# 4. 中心像素详细分析
print(f"\n4. 中心像素详细分析:")
cy_int, cx_int = H // 2, W // 2
print(f"   位置: ({cy_int}, {cx_int})")
print(f"   体渲染深度 (expected): {vol_depth[cy_int, cx_int]:.4f}")
print(f"   体渲染深度 (median): {vol_depth_med[cy_int, cx_int]:.4f}")
print(f"   几何深度: {geo_depth[cy_int, cx_int]:.4f}")

if geo_depth[cy_int, cx_int] > 0 and vol_depth[cy_int, cx_int] > 0:
    print(f"   比例: {vol_depth[cy_int, cx_int] / geo_depth[cy_int, cx_int]:.4f}")
