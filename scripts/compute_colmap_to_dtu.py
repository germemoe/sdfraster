#!/usr/bin/env python3
"""
计算 COLMAP 坐标系到 DTU 坐标系的变换矩阵

方法：使用相机位置进行配准
- 从 COLMAP 的 images.bin 提取相机位置
- 从 DTU 的 cameras.npz 提取相机位置（通过 world_mat @ scale_mat）
- 使用 Umeyama 算法计算刚体变换（带 scale）

用法:
    python scripts/compute_colmap_to_dtu.py --data_dir /path/to/DTU/scan37

输出:
    colmap_to_dtu.txt - 4x4 变换矩阵
"""
import argparse
import numpy as np
from pathlib import Path
import cv2


def load_K_Rt_from_P(P):
    """从投影矩阵 P 分解出内参 K 和位姿 [R|t]"""
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_dtu_camera_centers(cameras_npz_path):
    """从 DTU cameras.npz 提取相机中心位置（DTU 世界坐标系）"""
    camera_dict = np.load(cameras_npz_path)

    centers = []
    idx = 0
    while f'world_mat_{idx}' in camera_dict:
        world_mat = camera_dict[f'world_mat_{idx}']
        scale_mat = camera_dict[f'scale_mat_{idx}']

        # P = world_mat @ scale_mat
        P = world_mat @ scale_mat
        P = P[:3, :4]

        # 分解得到 c2w
        _, c2w = load_K_Rt_from_P(P)

        # 相机中心 = c2w 的平移部分
        # 但这是在归一化空间，需要用 scale_mat 变换回世界坐标
        cam_center_normalized = c2w[:3, 3]

        # 变换到 DTU 世界坐标系
        # 归一化空间 → 世界空间：应用 scale_mat
        cam_center_world = cam_center_normalized * scale_mat[0, 0] + scale_mat[:3, 3]

        centers.append(cam_center_world)
        idx += 1

    return np.array(centers)


def get_colmap_camera_centers(sparse_path):
    """从 COLMAP sparse 文件夹提取相机中心位置"""
    try:
        import pycolmap
        reconstruction = pycolmap.Reconstruction(sparse_path)

        centers = []
        # 按 image_id 排序
        for image_id in sorted(reconstruction.images.keys()):
            image = reconstruction.images[image_id]
            # COLMAP 的 cam_from_world，需要求逆得到 world_from_cam
            # 相机中心 = -R^T @ t
            R = image.rotation_matrix()
            t = image.tvec
            cam_center = -R.T @ t
            centers.append(cam_center)

        return np.array(centers)
    except ImportError:
        print("pycolmap not installed, trying to read binary files directly...")
        return read_colmap_cameras_binary(sparse_path)


def read_colmap_cameras_binary(sparse_path):
    """手动读取 COLMAP images.bin"""
    import struct

    images_path = Path(sparse_path) / 'images.bin'
    if not images_path.exists():
        raise FileNotFoundError(f"images.bin not found at {images_path}")

    centers = []
    with open(images_path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
            tx, ty, tz = struct.unpack('<ddd', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]

            # 读取 image name (null-terminated string)
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name += c

            # 读取 2D points
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2D * 24)  # x, y, point3D_id for each point

            # 四元数转旋转矩阵
            R = quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])

            # 相机中心 = -R^T @ t
            cam_center = -R.T @ t
            centers.append((image_id, cam_center))

    # 按 image_id 排序
    centers.sort(key=lambda x: x[0])
    return np.array([c[1] for c in centers])


def quat_to_rotmat(qw, qx, qy, qz):
    """四元数转旋转矩阵"""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R


def umeyama_alignment(src, dst, with_scale=True):
    """
    Umeyama 算法：计算从 src 到 dst 的最优刚体变换（可选 scale）

    Args:
        src: (N, 3) 源点云
        dst: (N, 3) 目标点云
        with_scale: 是否计算 scale

    Returns:
        T: (4, 4) 变换矩阵，使得 dst ≈ T @ src
        scale: float, 缩放因子
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
    """
    assert src.shape == dst.shape
    n, dim = src.shape

    # 计算质心
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # 去质心
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # 计算方差
    src_var = np.sum(src_centered ** 2) / n

    # 计算协方差矩阵
    H = src_centered.T @ dst_centered / n

    # SVD 分解
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转
    R = Vt.T @ U.T

    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算 scale
    if with_scale:
        scale = np.sum(S) / src_var
    else:
        scale = 1.0

    # 计算平移
    t = dst_mean - scale * R @ src_mean

    # 构建 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return T, scale, R, t


def main():
    parser = argparse.ArgumentParser(description='计算 COLMAP → DTU 坐标变换')
    parser.add_argument('--data_dir', type=str, required=True, help='DTU scene 目录（包含 cameras.npz 和 sparse/0/）')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（默认为 data_dir/colmap_to_dtu.txt）')
    parser.add_argument('--no_scale', action='store_true', help='不计算 scale（仅刚体变换）')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 检查必要文件
    cameras_npz = data_dir / 'cameras.npz'
    sparse_dir = data_dir / 'sparse' / '0'

    if not cameras_npz.exists():
        print(f"Error: cameras.npz not found at {cameras_npz}")
        return

    if not sparse_dir.exists():
        print(f"Error: sparse/0/ not found at {sparse_dir}")
        return

    # 提取相机中心
    print("Loading DTU camera centers...")
    dtu_centers = get_dtu_camera_centers(cameras_npz)
    print(f"  Found {len(dtu_centers)} cameras")
    print(f"  DTU center range: {dtu_centers.min(axis=0)} to {dtu_centers.max(axis=0)}")

    print("\nLoading COLMAP camera centers...")
    colmap_centers = get_colmap_camera_centers(sparse_dir)
    print(f"  Found {len(colmap_centers)} cameras")
    print(f"  COLMAP center range: {colmap_centers.min(axis=0)} to {colmap_centers.max(axis=0)}")

    # 检查相机数量是否匹配
    if len(dtu_centers) != len(colmap_centers):
        print(f"\nWarning: camera count mismatch! DTU={len(dtu_centers)}, COLMAP={len(colmap_centers)}")
        # 使用较小的数量
        n = min(len(dtu_centers), len(colmap_centers))
        dtu_centers = dtu_centers[:n]
        colmap_centers = colmap_centers[:n]
        print(f"  Using first {n} cameras")

    # 计算变换
    print("\nComputing Umeyama alignment (COLMAP → DTU)...")
    T, scale, R, t = umeyama_alignment(colmap_centers, dtu_centers, with_scale=not args.no_scale)

    # 验证变换
    colmap_transformed = (T[:3, :3] @ colmap_centers.T).T + T[:3, 3]
    error = np.linalg.norm(colmap_transformed - dtu_centers, axis=1)
    print(f"\nAlignment results:")
    print(f"  Scale: {scale:.6f}")
    print(f"  Translation: {t}")
    print(f"  Mean error: {error.mean():.4f}")
    print(f"  Max error: {error.max():.4f}")

    # 保存变换矩阵
    output_path = args.output if args.output else data_dir / 'colmap_to_dtu.txt'
    np.savetxt(output_path, T, fmt='%.10f')
    print(f"\nSaved transformation matrix to: {output_path}")
    print("\nTransformation matrix:")
    print(T)


if __name__ == '__main__':
    main()
