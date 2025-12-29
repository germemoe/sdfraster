#!/usr/bin/env python3
"""
从训练完成的 checkpoint 中使用 Marching Cubes 提取 mesh

用法:
    python scripts/extract_mesh_mc.py --ckpt_path output/sdf/DTU/test/mesh_depth
    python scripts/extract_mesh_mc.py --ckpt_path output/sdf/DTU/test/mesh_depth --resolution 512
    python scripts/extract_mesh_mc.py --ckpt_path output/sdf/DTU/test/mesh_depth --output mesh.ply --keep_lcc

    # 使用初始点云的 bounding box
    python scripts/extract_mesh_mc.py --ckpt_path output/test --pcd_path data/scan37/sparse/0/points3D.ply --bounds_scale 1.2
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import mcubes
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from models.frozen import FrozenTetModel


@torch.no_grad()
def extract_mesh_from_sdf(
    sdf_func,
    bounds,
    resolution=1024,
    block_res=64,
    threshold=0.0,
    filter_lcc=False,
    device='cuda'
):
    """
    使用 Marching Cubes 从 SDF 函数提取 mesh

    Args:
        sdf_func: SDF 查询函数，输入 (N, 3) 返回 (N,)
        bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        resolution: Marching Cubes 分辨率
        block_res: 分块大小（避免 OOM）
        threshold: SDF 阈值（默认 0）
        filter_lcc: 是否只保留最大连通分量
        device: 计算设备

    Returns:
        trimesh.Trimesh: 提取的 mesh
    """
    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) = bounds

    # 创建网格
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)

    # 计算步长
    intv_x = (x_max - x_min) / (resolution - 1)
    intv_y = (y_max - y_min) / (resolution - 1)
    intv_z = (z_max - z_min) / (resolution - 1)

    # 分块计算 SDF（避免 OOM）
    sdf_volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    num_blocks_x = int(np.ceil(resolution / block_res))
    num_blocks_y = int(np.ceil(resolution / block_res))
    num_blocks_z = int(np.ceil(resolution / block_res))

    total_blocks = num_blocks_x * num_blocks_y * num_blocks_z
    print(f"Extracting SDF at resolution {resolution}x{resolution}x{resolution}")
    print(f"Processing {total_blocks} blocks...")

    pbar = tqdm(total=total_blocks, desc="Querying SDF")

    for bx in range(num_blocks_x):
        for by in range(num_blocks_y):
            for bz in range(num_blocks_z):
                # 计算块范围
                xi = bx * block_res
                yi = by * block_res
                zi = bz * block_res
                xe = min(xi + block_res, resolution)
                ye = min(yi + block_res, resolution)
                ze = min(zi + block_res, resolution)

                # 创建块内的坐标网格
                xx, yy, zz = np.meshgrid(
                    x[xi:xe], y[yi:ye], z[zi:ze], indexing='ij'
                )
                xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

                # 查询 SDF
                xyz_tensor = torch.from_numpy(xyz).float().to(device)
                sdf = sdf_func(xyz_tensor).cpu().numpy()

                # 填充到体积中
                sdf_volume[xi:xe, yi:ye, zi:ze] = sdf.reshape(xe-xi, ye-yi, ze-zi)

                pbar.update(1)

    pbar.close()

    # Marching Cubes
    print("Running Marching Cubes...")
    vertices, faces = mcubes.marching_cubes(sdf_volume, threshold)

    if vertices.shape[0] == 0:
        print("Warning: No surface extracted!")
        return trimesh.Trimesh()

    # 转换顶点坐标到世界坐标
    vertices[:, 0] = vertices[:, 0] * intv_x + x_min
    vertices[:, 1] = vertices[:, 1] * intv_y + y_min
    vertices[:, 2] = vertices[:, 2] * intv_z + z_min

    mesh = trimesh.Trimesh(vertices, faces)

    # 过滤最大连通分量
    if filter_lcc:
        print("Filtering largest connected component...")
        components = mesh.split(only_watertight=False)
        if len(components) > 0:
            areas = np.array([c.area for c in components], dtype=float)
            mesh = components[areas.argmax()]
            print(f"Kept largest component with {len(mesh.vertices)} vertices")

    return mesh


def main():
    parser = argparse.ArgumentParser(description='从 checkpoint 使用 Marching Cubes 提取 mesh')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Checkpoint 目录路径（包含 ckpt.pth 和 config.json）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（默认为 ckpt_path/extracted_mesh_mc.ply）')
    parser.add_argument('--resolution', type=int, default=2048,
                        help='Marching Cubes 分辨率（默认 256）')
    parser.add_argument('--block_res', type=int, default=128,
                        help='分块大小（默认 64）')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='SDF 阈值（默认 0.0）')
    parser.add_argument('--keep_lcc', action='store_true',
                        help='只保留最大连通分量')
    parser.add_argument('--bounds_scale', type=float, default=1.0,
                        help='边界缩放因子。使用 --pcd_path 时：1.0=原始大小，1.2=扩大20%%；否则：作为 margin 比例（默认 1.0）')
    parser.add_argument('--frozen', action='store_true',
                        help='加载 FrozenTetModel')
    parser.add_argument('--world_coords', action='store_true',
                        help='输出世界坐标系的 mesh（应用逆变换）')
    parser.add_argument('--pcd_path', type=str, default=None,
                        help='初始点云路径，指定后使用点云的 bounding box 初始化 MC grid')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.ckpt_path)

    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"Device: {device}")

    # 加载模型
    if args.frozen:
        model = FrozenTetModel.load_ckpt(ckpt_path, device)
    else:
        model = Model.load_ckpt(ckpt_path, device)

    model.eval()

    print(f"Model loaded successfully")
    print(f"  Total vertices: {model.vertices.shape[0]}")
    print(f"  Internal vertices: {model.num_int_verts}")
    print(f"  External vertices: {model.ext_vertices.shape[0]}")
    print(f"  Tetrahedra: {model.indices.shape[0]}")
    print(f"  Scene scaling: {model.scene_scaling}")
    print(f"  Center: {model.center}")

    # 计算边界
    if args.pcd_path is not None:
        # 使用初始点云的 bounding box
        print(f"Loading point cloud from: {args.pcd_path}")
        pcd = trimesh.load(args.pcd_path)
        if hasattr(pcd, 'vertices'):
            pcd_vertices = np.array(pcd.vertices)
        else:
            pcd_vertices = np.array(pcd)

        # 应用训练时的 transform（将点云变换到训练坐标系）
        transform_file = ckpt_path / "transform.txt"
        if transform_file.exists():
            print("Applying transform to point cloud...")
            transform = np.loadtxt(str(transform_file))
            pcd_hom = np.hstack([pcd_vertices, np.ones((pcd_vertices.shape[0], 1))])
            pcd_vertices = (transform @ pcd_hom.T).T[:, :3]

        v_min = pcd_vertices.min(axis=0)
        v_max = pcd_vertices.max(axis=0)
        print(f"Point cloud vertices: {pcd_vertices.shape[0]}")
        print(f"Point cloud bounds: min={v_min}, max={v_max}")

        # bounds_scale 作为缩放因子（1.0 = 原始大小，1.2 = 扩大 20%）
        center = (v_min + v_max) / 2
        half_size = (v_max - v_min) / 2 * args.bounds_scale
        v_min = center - half_size
        v_max = center + half_size
    else:
        # 使用内部顶点的 bounding box（排除外部的 ext_vertices）
        internal_vertices = model.vertices[:model.num_int_verts].detach()
        v_min = internal_vertices.min(dim=0).values.cpu().numpy()
        v_max = internal_vertices.max(dim=0).values.cpu().numpy()

        # 添加 margin
        margin = (v_max - v_min) * 0.1 * args.bounds_scale
        v_min = v_min - margin
        v_max = v_max + margin

    bounds = [
        [v_min[0], v_max[0]],
        [v_min[1], v_max[1]],
        [v_min[2], v_max[2]],
    ]

    print(f"Bounds: {bounds}")

    # 定义 SDF 查询函数
    def sdf_func(xyz):
        return model.query_vert_sdf(xyz)

    # 提取 mesh
    mesh = extract_mesh_from_sdf(
        sdf_func=sdf_func,
        bounds=bounds,
        resolution=args.resolution,
        block_res=args.block_res,
        threshold=args.threshold,
        filter_lcc=args.keep_lcc,
        device=device
    )

    if mesh.vertices.shape[0] == 0:
        print("No mesh extracted!")
        return

    # 转换到世界坐标系（如果需要）
    if args.world_coords:
        transform_file = ckpt_path / "transform.txt"
        if transform_file.exists():
            print("Transforming to world coordinates...")
            transform = np.loadtxt(str(transform_file))
            inv_transform = np.linalg.inv(transform)

            # 应用逆变换
            verts_hom = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])
            verts_world = (inv_transform @ verts_hom.T).T[:, :3]
            mesh.vertices = verts_world
        else:
            print(f"Warning: transform.txt not found at {transform_file}, skipping world coord transform")

    # 输出路径
    if args.output is None:
        output_path = ckpt_path / "extracted_mesh_mc.ply"
    else:
        output_path = ckpt_path / Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存 mesh
    mesh.export(str(output_path))

    print(f"\nMesh extracted successfully!")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Saved to: {output_path}")


if __name__ == '__main__':
    main()
