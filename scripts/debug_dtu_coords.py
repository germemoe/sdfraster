#!/usr/bin/env python3
"""
调试 DTU 坐标系问题：检查 mesh 和 GT 点云的坐标范围

用法:
    python scripts/debug_dtu_coords.py --mesh output/test/final_mesh.ply --dtu_root /path/to/DTU --scan_id 37
"""
import argparse
import numpy as np
import trimesh
from scipy.io import loadmat
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='调试 DTU 坐标系')
    parser.add_argument('--mesh', type=str, required=True, help='要评估的 mesh 路径')
    parser.add_argument('--dtu_root', type=str, required=True, help='DTU 数据集根目录')
    parser.add_argument('--scan_id', type=str, default='37', help='scan ID')
    parser.add_argument('--transform', type=str, default=None, help='训练时的 transform.txt 路径（可选）')
    args = parser.parse_args()

    scan_id = args.scan_id
    dtu_root = Path(args.dtu_root)

    # 1. 加载 mesh
    print("=" * 60)
    print(f"[1] Loading mesh: {args.mesh}")
    mesh = trimesh.load(args.mesh)
    mesh_verts = np.asarray(mesh.vertices)
    print(f"    Vertices: {mesh_verts.shape[0]}")
    print(f"    Bounds: min={mesh_verts.min(axis=0)}, max={mesh_verts.max(axis=0)}")
    print(f"    Center: {mesh_verts.mean(axis=0)}")

    # 2. 加载 DTU ObsMask 的 bounding box
    print("\n" + "=" * 60)
    obs_mask_path = dtu_root / 'ObsMask' / f'ObsMask{scan_id}_10.mat'
    print(f"[2] Loading ObsMask: {obs_mask_path}")
    if obs_mask_path.exists():
        obs_mask_file = loadmat(str(obs_mask_path))
        BB = obs_mask_file['BB'].astype(np.float32)
        Res = obs_mask_file['Res']
        ObsMask = obs_mask_file['ObsMask']
        print(f"    BB (bounding box): min={BB[0]}, max={BB[1]}")
        print(f"    Res (resolution): {Res}")
        print(f"    ObsMask shape: {ObsMask.shape}")

        # 检查 mesh 点有多少在 BB 内
        patch = 60  # 与 eval.py 一致
        inbound = ((mesh_verts >= BB[0] - patch) & (mesh_verts < BB[1] + patch * 2)).sum(axis=-1) == 3
        print(f"    Mesh points in BB (with patch={patch}): {inbound.sum()} / {len(mesh_verts)}")
    else:
        print(f"    [WARNING] ObsMask not found!")
        BB = None

    # 3. 加载 DTU GT 点云
    print("\n" + "=" * 60)
    gt_path = dtu_root / 'Points' / 'stl' / f'stl{int(scan_id):03d}_total.ply'
    print(f"[3] Loading GT point cloud: {gt_path}")
    if gt_path.exists():
        gt_pcd = trimesh.load(str(gt_path))
        if hasattr(gt_pcd, 'vertices'):
            gt_verts = np.asarray(gt_pcd.vertices)
        else:
            gt_verts = np.asarray(gt_pcd)
        print(f"    Points: {gt_verts.shape[0]}")
        print(f"    Bounds: min={gt_verts.min(axis=0)}, max={gt_verts.max(axis=0)}")
        print(f"    Center: {gt_verts.mean(axis=0)}")
    else:
        print(f"    [WARNING] GT point cloud not found!")
        gt_verts = None

    # 4. 加载 cameras.npz（如果存在）
    print("\n" + "=" * 60)
    cameras_path = dtu_root / f'scan{scan_id}' / 'cameras.npz'
    print(f"[4] Loading cameras.npz: {cameras_path}")
    if cameras_path.exists():
        camera_dict = np.load(str(cameras_path))
        scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
        print(f"    scale_mat[0,0] (scale): {scale_mat[0, 0]}")
        print(f"    scale_mat[:3,3] (translation): {scale_mat[:3, 3]}")

        # 尝试应用 scale_mat 变换
        mesh_verts_scaled = mesh_verts * scale_mat[0, 0] + scale_mat[:3, 3]
        print(f"\n    After scale_mat transform:")
        print(f"    Bounds: min={mesh_verts_scaled.min(axis=0)}, max={mesh_verts_scaled.max(axis=0)}")

        if BB is not None:
            inbound_scaled = ((mesh_verts_scaled >= BB[0] - patch) & (mesh_verts_scaled < BB[1] + patch * 2)).sum(axis=-1) == 3
            print(f"    Mesh points in BB (after scale_mat): {inbound_scaled.sum()} / {len(mesh_verts)}")
    else:
        print(f"    [WARNING] cameras.npz not found!")

    # 5. 加载训练时的 transform（如果提供）
    print("\n" + "=" * 60)
    if args.transform:
        transform_path = Path(args.transform)
    else:
        # 尝试从 mesh 路径推断
        transform_path = Path(args.mesh).parent / 'transform.txt'

    print(f"[5] Loading transform: {transform_path}")
    if transform_path.exists():
        transform = np.loadtxt(str(transform_path))
        print(f"    Transform matrix:\n{transform}")

        # 尝试应用逆变换（检查是否已经应用过）
        inv_transform = np.linalg.inv(transform)
        mesh_verts_hom = np.hstack([mesh_verts, np.ones((mesh_verts.shape[0], 1))])
        mesh_verts_inv = (inv_transform @ mesh_verts_hom.T).T[:, :3]
        print(f"\n    After inverse transform (back to training coords):")
        print(f"    Bounds: min={mesh_verts_inv.min(axis=0)}, max={mesh_verts_inv.max(axis=0)}")
    else:
        print(f"    [WARNING] transform.txt not found!")

    # 6. 诊断建议
    print("\n" + "=" * 60)
    print("[6] 诊断建议:")
    if BB is not None and gt_verts is not None:
        mesh_center = mesh_verts.mean(axis=0)
        gt_center = gt_verts.mean(axis=0)
        bb_center = (BB[0] + BB[1]) / 2

        dist_mesh_to_gt = np.linalg.norm(mesh_center - gt_center)
        dist_mesh_to_bb = np.linalg.norm(mesh_center - bb_center)

        print(f"    Mesh center 到 GT center 距离: {dist_mesh_to_gt:.2f}")
        print(f"    Mesh center 到 BB center 距离: {dist_mesh_to_bb:.2f}")

        if dist_mesh_to_gt > 100 or dist_mesh_to_bb > 100:
            print("\n    [!] 坐标系明显不匹配！可能的解决方案：")
            print("        1. 如果使用 COLMAP 格式 DTU，尝试添加 --apply_scale_transform")
            print("        2. 检查训练数据加载是否正确")
            print("        3. 检查 mesh 保存时的坐标变换是否正确")
        else:
            print("\n    坐标系看起来基本对齐")


if __name__ == '__main__':
    main()
