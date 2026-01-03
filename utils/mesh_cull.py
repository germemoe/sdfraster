"""
Mesh Culling and Scale Alignment Utilities (参考 MILo)

用于 DTU 评估前的 mesh 预处理：
1. cull_mesh: 使用相机 mask 过滤 mesh 顶点
2. best_fit_transform: 计算最优刚体变换对齐相机位置
"""

import numpy as np
import torch
import torch.nn.functional as F
import math

try:
    from skimage.morphology import binary_dilation, disk
except ImportError:
    print("Warning: skimage not available, cull_mesh will not work")
    binary_dilation = None
    disk = None


def fov2focal(fov, pixels):
    """将 FOV 转换为焦距"""
    return pixels / (2 * math.tan(fov / 2))


def best_fit_transform(A, B):
    """
    计算将点集 A 对齐到点集 B 的最优刚体变换（最小二乘）

    Args:
        A: (N, m) 源点集
        B: (N, m) 目标点集

    Returns:
        T: (m+1, m+1) 齐次变换矩阵
        R: (m, m) 旋转矩阵
        t: (m,) 平移向量
    """
    assert A.shape == B.shape

    m = A.shape[1]

    # 计算质心
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # 旋转矩阵 (SVD)
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 平移向量
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # 齐次变换矩阵
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def cull_mesh_with_cameras(mesh_verts, mesh_faces, cameras, dilation_radius=6):
    """
    使用相机 mask 过滤 mesh 顶点（参考 MILo）

    只保留在所有相机视角中都可见（在 mask 内）的顶点。

    Args:
        mesh_verts: (N, 3) numpy array, mesh 顶点坐标（训练坐标系）
        mesh_faces: (M, 3) numpy array, mesh 面片索引
        cameras: 相机列表，每个相机需要有:
            - world_view_transform: W2V.T 矩阵
            - fovx, fovy: 视场角
            - image_width, image_height: 图像尺寸
            - gt_alpha_mask: GT mask (1, H, W) 或 (H, W)
        dilation_radius: mask 膨胀半径（默认 6，参考 MILo）

    Returns:
        culled_verts: 过滤后的顶点
        culled_faces: 过滤后的面片
    """
    if binary_dilation is None:
        print("Warning: skimage not available, skipping mesh culling")
        return mesh_verts, mesh_faces

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 转换顶点为齐次坐标
    vertices = torch.from_numpy(mesh_verts).float().to(device)
    vertices = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    vertices = vertices.T  # (4, N)

    sampled_masks = []

    for camera in cameras:
        # 获取相机参数
        w2v = camera.world_view_transform.T.to(device)  # (4, 4)

        # 获取 mask
        gt_mask = getattr(camera, 'gt_alpha_mask', None)
        if gt_mask is None:
            # 如果没有 mask，使用全 1 mask
            H = camera.image_height
            W = camera.image_width
            mask_np = np.ones((H, W), dtype=np.float32)
        else:
            mask_np = gt_mask.squeeze().cpu().numpy().astype(np.float32)
            if mask_np.max() > 1:
                mask_np = mask_np / 255.0

        # 计算内参矩阵
        fx = fov2focal(camera.fovx, camera.image_width)
        fy = fov2focal(camera.fovy, camera.image_height)
        W = camera.image_width
        H = camera.image_height

        intrinsic = torch.eye(4, device=device)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = W / 2.0
        intrinsic[1, 2] = H / 2.0

        with torch.no_grad():
            # 投影顶点到相机平面
            cam_points = intrinsic @ w2v @ vertices  # (4, N)
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.T  # (N, 2)

            # 归一化到 [-1, 1]
            pix_coords[..., 0] /= (W - 1)
            pix_coords[..., 1] /= (H - 1)
            pix_coords = (pix_coords - 0.5) * 2

            # 检查是否在图像范围内
            valid = ((pix_coords > -1.0) & (pix_coords < 1.0)).all(dim=-1).float()

            # 膨胀 mask（参考 MILo: disk(6)）
            mask_dilated = binary_dilation(mask_np, disk(dilation_radius))
            mask_tensor = torch.from_numpy(mask_dilated.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            # 采样 mask
            sampled_mask = F.grid_sample(
                mask_tensor,
                pix_coords[None, None],  # (1, 1, N, 2)
                mode='nearest',
                padding_mode='zeros',
                align_corners=True
            )[0, 0, 0]  # (N,)

            # 不在图像范围内的顶点也视为有效（避免被错误剔除）
            sampled_mask = sampled_mask + (1.0 - valid)

            sampled_masks.append(sampled_mask)

    # 合并所有相机的 mask
    sampled_masks = torch.stack(sampled_masks, dim=-1)  # (N, num_cameras)

    # 只保留在所有相机中都可见的顶点
    vert_mask = (sampled_masks > 0.0).all(dim=-1).cpu().numpy()

    # 过滤面片（只保留三个顶点都可见的面片）
    face_mask = vert_mask[mesh_faces].all(axis=1)

    # 重建顶点和面片
    new_vert_indices = np.cumsum(vert_mask) - 1
    culled_verts = mesh_verts[vert_mask]
    culled_faces = new_vert_indices[mesh_faces[face_mask]]

    print(f"[Cull] Vertices: {mesh_verts.shape[0]} -> {culled_verts.shape[0]}")
    print(f"[Cull] Faces: {mesh_faces.shape[0]} -> {culled_faces.shape[0]}")

    return culled_verts, culled_faces


def load_dtu_calibration_cameras(dtu_path):
    """
    加载 DTU 标定相机位置

    Args:
        dtu_path: DTU 数据集根目录（包含 Calibration 文件夹）

    Returns:
        camera_positions: (64, 3) numpy array, 相机位置
    """
    import os
    from os import path
    import cv2

    camera_positions = []
    for i in range(1, 65):
        fname = path.join(dtu_path, f'Calibration/cal18/pos_{i:03d}.txt')
        if not path.exists(fname):
            break

        projection = np.loadtxt(fname, dtype=np.float32)
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        camera_positions.append(pose[:3, 3])

    return np.array(camera_positions)


def compute_scale_alignment(train_cameras, dtu_camera_positions):
    """
    计算训练相机位置与 DTU 标定相机位置之间的对齐变换（参考 MILo）

    Args:
        train_cameras: 训练相机列表
        dtu_camera_positions: (N, 3) DTU 标定相机位置

    Returns:
        scale_ratio: 缩放比例 (scale_gt / scale_train)
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
    """
    # 提取训练相机位置
    train_positions = []
    for cam in train_cameras:
        c2w = cam.world_view_transform.T.inverse()
        train_positions.append(c2w[:3, 3].cpu().numpy())
    train_positions = np.array(train_positions)

    # 只使用相同数量的相机
    num_cams = min(len(train_positions), len(dtu_camera_positions))
    train_positions = train_positions[:num_cams]
    gt_positions = dtu_camera_positions[:num_cams]

    # 计算缩放比例
    scale_train = np.linalg.norm(train_positions - train_positions.mean(axis=0), axis=1).mean()
    scale_gt = np.linalg.norm(gt_positions - gt_positions.mean(axis=0), axis=1).mean()
    scale_ratio = scale_gt / scale_train

    # 缩放后计算最优刚体变换
    scaled_train = train_positions * scale_ratio
    _, R, t = best_fit_transform(scaled_train, gt_positions)

    return scale_ratio, R, t


def align_mesh_to_dtu(mesh_verts, train_cameras, dtu_path):
    """
    将 mesh 顶点从训练坐标系对齐到 DTU 世界坐标系（参考 MILo）

    流程：
    1. 加载 DTU 标定相机位置
    2. 计算缩放和刚体变换
    3. 变换 mesh 顶点

    Args:
        mesh_verts: (N, 3) mesh 顶点（训练坐标系）
        train_cameras: 训练相机列表
        dtu_path: DTU 数据集路径

    Returns:
        aligned_verts: (N, 3) 对齐后的顶点（DTU 世界坐标系）
    """
    import os

    # 检查 DTU 标定文件是否存在
    calib_path = os.path.join(dtu_path, 'Calibration/cal18/pos_001.txt')
    if not os.path.exists(calib_path):
        print(f"[Align] DTU calibration not found: {calib_path}")
        print("[Align] Skipping scale alignment, using identity transform")
        return mesh_verts

    # 加载 DTU 标定相机位置
    gt_positions = load_dtu_calibration_cameras(dtu_path)

    # 计算对齐变换
    scale_ratio, R, t = compute_scale_alignment(train_cameras, gt_positions)

    print(f"[Align] Scale ratio: {scale_ratio:.4f}")
    print(f"[Align] Translation: {t}")

    # 变换 mesh 顶点
    aligned_verts = mesh_verts * scale_ratio
    aligned_verts = aligned_verts @ R.T + t

    return aligned_verts
