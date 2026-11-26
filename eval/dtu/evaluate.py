"""
DTU Mesh Evaluation

直接复用MILo的评估代码，确保正确性。
Reference: MILo/milo/eval/dtu/evaluate_dtu_mesh.py

坐标对齐方法（MILo方法B）：
1. 获取训练相机位置（训练坐标系）
2. 获取DTU标定相机位置（DTU世界坐标系）
3. 计算缩放 + 刚体变换
4. 应用到mesh顶点

这种方法不依赖于 scale_mat 或 inv_transform，而是直接通过对应点对齐。
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import trimesh
import math
from os import path
from skimage.morphology import binary_dilation, disk


# ========== 直接从 MILo 复制的函数 ==========

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def load_dtu_camera(DTU):
    """Load DTU calibration cameras from Calibration/cal18/pos_*.txt"""
    camtoworlds = []
    for i in range(1, 64+1):
        fname = path.join(DTU, f'Calibration/cal18/pos_{i:03d}.txt')
        if not os.path.exists(fname):
            continue

        projection = np.loadtxt(fname, dtype=np.float32)

        # Decompose projection matrix into pose and camera matrix.
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        pose = pose[:3]
        camtoworlds.append(pose)
    return camtoworlds


def load_scale_mat(dataset_path, scan_id=None):
    """
    从 cameras.npz 加载 scale_mat，用于将归一化坐标变换到DTU世界坐标系。

    注意：是 scale_mat 不是 scale_mat_inv！
    scale_mat 把归一化坐标（约[-1,1]）变换到DTU世界坐标（约几百mm）。

    Args:
        dataset_path: scan目录路径（包含 cameras.npz）
        scan_id: 未使用，仅为兼容性保留

    Returns:
        scale_mat: (4, 4) numpy array, 或 None 如果不存在
    """
    cameras_npz_path = path.join(dataset_path, 'cameras.npz')
    if not os.path.exists(cameras_npz_path):
        return None

    data = np.load(cameras_npz_path)
    # 使用第0个相机的 scale_mat（所有相机的 scale_mat 应该相同）
    if 'scale_mat_0' in data:
        return data['scale_mat_0'].astype(np.float64)
    return None


def load_pca_transform(model_output_path):
    """
    从模型输出目录加载 PCA 变换矩阵。

    训练时会对相机位置做 PCA 变换，transform.txt 保存了这个变换。
    评估时需要用逆变换把训练坐标变回 COLMAP/归一化坐标。

    Args:
        model_output_path: 模型输出目录（包含 transform.txt）

    Returns:
        pca_transform: (4, 4) numpy array, 或 None 如果不存在
    """
    transform_path = path.join(model_output_path, 'transform.txt')
    if not os.path.exists(transform_path):
        return None
    return np.loadtxt(transform_path).astype(np.float64)


def align_mesh_with_transforms(mesh_vertices, pca_transform=None, scale_mat=None):
    """
    将mesh从训练坐标变换到DTU世界坐标。

    变换链：训练坐标 --[inv(pca_transform)]--> 归一化坐标 --[scale_mat]--> DTU世界坐标

    Args:
        mesh_vertices: (N, 3) numpy array, 训练坐标系下的顶点
        pca_transform: (4, 4) PCA变换矩阵，如果为None则跳过
        scale_mat: (4, 4) scale变换矩阵，如果为None则跳过

    Returns:
        aligned_vertices: (N, 3) DTU世界坐标系下的顶点
    """
    N = mesh_vertices.shape[0]
    vertices = mesh_vertices.copy()

    # Step 1: 应用 PCA 逆变换（训练坐标 -> 归一化坐标）
    if pca_transform is not None:
        pca_transform_inv = np.linalg.inv(pca_transform)
        vertices_hom = np.hstack([vertices, np.ones((N, 1))])
        vertices = (pca_transform_inv @ vertices_hom.T).T[:, :3]

    # Step 2: 应用 scale_mat（归一化坐标 -> DTU世界坐标）
    if scale_mat is not None:
        vertices_hom = np.hstack([vertices, np.ones((N, 1))])
        vertices = (scale_mat @ vertices_hom.T).T[:, :3]

    return vertices


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def cull_mesh(cameras, mesh, dilation_radius=6):
    """
    Cull mesh vertices using camera masks.

    直接复用MILo的实现，确保和MILo行为一致。

    Args:
        cameras: 训练相机列表
        mesh: trimesh.Trimesh对象
        dilation_radius: mask膨胀半径（MILo默认6）

    Returns:
        culled mesh
    """
    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []

    for camera in cameras:
        c2w = (camera.world_view_transform.T).inverse()
        w2c = torch.inverse(c2w).cuda()

        # 兼容不同的mask属性名称
        mask = getattr(camera, 'gt_mask', None)
        if mask is None:
            mask = getattr(camera, 'gt_alpha_mask', None)

        # 处理没有mask的情况
        if mask is None:
            H, W = camera.image_height, camera.image_width
            mask = torch.ones(1, H, W).cuda()
        else:
            # 确保 mask 在 GPU 上且有正确的形状
            if not mask.is_cuda:
                mask = mask.cuda()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)

        fx = fov2focal(camera.fovx, camera.image_width)
        fy = fov2focal(camera.fovy, camera.image_height)
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = camera.image_width / 2.
        intrinsic[1, 2] = camera.image_height / 2.
        intrinsic = intrinsic.cuda()

        W, H = camera.image_width, camera.image_height

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1.) & (pix_coords < 1.)).all(dim=-1).float()

            # dialate mask similar to unisurf
            maski = mask[0, :, :].cpu().numpy().astype(np.float32)
            if maski.max() > 1:
                maski = maski / 255.
            maski = torch.from_numpy(binary_dilation(maski, disk(dilation_radius))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)

            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)

    # filter
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    return mesh


def align_mesh_to_dtu(mesh_vertices, train_cameras, dtu_cameras):
    """
    将mesh从训练坐标系对齐到DTU世界坐标系。

    使用MILo的方法：通过训练相机位置和DTU标定相机位置计算对齐变换。

    Args:
        mesh_vertices: (N, 3) numpy array, 训练坐标系下的顶点
        train_cameras: 训练相机列表
        dtu_cameras: DTU标定相机列表 (从load_dtu_camera返回)

    Returns:
        aligned_vertices: (N, 3) DTU世界坐标系下的顶点
        scale: 缩放因子
        R: 旋转矩阵
        t: 平移向量
    """
    # 获取DTU相机位置
    gt_points = np.array([cam[:, 3] for cam in dtu_cameras])

    # 获取训练相机位置
    points = []
    for cam in train_cameras:
        c2w = (cam.world_view_transform.T).inverse()
        points.append(c2w[:3, 3].cpu().numpy())
    points = np.array(points)

    # 匹配数量
    gt_points = gt_points[:points.shape[0]]

    # align the scale of two point clouds
    scale_points = np.linalg.norm(points - points.mean(axis=0), axis=1).mean()
    scale_gt_points = np.linalg.norm(gt_points - gt_points.mean(axis=0), axis=1).mean()

    scale = scale_gt_points / scale_points
    points_scaled = points * scale
    _, R, t = best_fit_transform(points_scaled, gt_points)

    # 应用变换到mesh顶点
    aligned_vertices = mesh_vertices * scale
    aligned_vertices = aligned_vertices @ R.T + t

    return aligned_vertices, scale, R, t


def evaluate_dtu_mesh(
    mesh_path,
    train_cameras,
    dtu_path,
    scan_id,
    output_dir=None,
    cull=True,
    dilation_radius=6,
    dataset_path=None,
    model_output_path=None,
    seed=None,
    downsample_method="batch",
    cpu_workers=None,
    write_visualizations=True,
):
    """
    完整的DTU mesh评估流程。

    坐标变换链：训练坐标 --[inv(pca_transform)]--> 归一化坐标 --[scale_mat]--> DTU世界坐标

    支持两种对齐方法：
    1. Transform 方法（推荐）: 使用 transform.txt + cameras.npz 中的 scale_mat
    2. Calibration 方法（MILo）: 使用 DTU Calibration 相机位置对齐（fallback）

    流程：
    1. cull_mesh（训练坐标系下）
    2. 对齐到DTU世界坐标系
    3. 调用eval_core评估

    Args:
        mesh_path: 输入mesh路径（训练坐标系）
        train_cameras: 训练相机列表
        dtu_path: DTU数据集根目录（包含 ObsMask/, Points/）
        scan_id: DTU scan ID
        output_dir: 输出目录
        cull: 是否进行cull_mesh
        dilation_radius: mask膨胀半径
        dataset_path: 数据集路径（包含 cameras.npz），用于 scale_mat
        model_output_path: 模型输出目录（包含 transform.txt），用于 PCA 变换
        cpu_workers: 采样和近邻 worker 上限；None 保留原自动 worker 行为

    Returns:
        评估结果字典
    """
    from eval.dtu.eval_core import evaluate_mesh as eval_core
    if type(write_visualizations) is not bool:
        raise TypeError("write_visualizations must be a bool")

    if output_dir is None:
        output_dir = os.path.dirname(mesh_path)
    os.makedirs(output_dir, exist_ok=True)

    # 加载mesh
    print(f"[DTU-EVAL] Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    print(f"    Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # Cull mesh
    if cull:
        print(f"[DTU-EVAL] Culling mesh (dilation={dilation_radius})...")
        mesh = cull_mesh(train_cameras, mesh, dilation_radius=dilation_radius)
        if write_visualizations:
            culled_path = os.path.join(output_dir, "mesh_culled.ply")
            mesh.export(culled_path)
        print(f"    After cull: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        if write_visualizations:
            print(f"    Saved: {culled_path}")

    # 加载变换矩阵
    pca_transform = None
    scale_mat = None

    if model_output_path is not None:
        pca_transform = load_pca_transform(model_output_path)
        if pca_transform is not None:
            print(f"[DTU-EVAL] Loaded PCA transform from {model_output_path}/transform.txt")

    if dataset_path is not None:
        scale_mat = load_scale_mat(dataset_path)
        if scale_mat is not None:
            print(f"[DTU-EVAL] Loaded scale_mat from {dataset_path}/cameras.npz")

    # 方法1: 使用 transform.txt + scale_mat 对齐（推荐）
    if pca_transform is not None or scale_mat is not None:
        print(f"[DTU-EVAL] Aligning mesh using transforms...")
        aligned_verts = align_mesh_with_transforms(
            np.asarray(mesh.vertices),
            pca_transform=pca_transform,
            scale_mat=scale_mat
        )
        if pca_transform is not None and scale_mat is not None:
            print(f"    Applied: inv(pca_transform) -> scale_mat")
        elif pca_transform is not None:
            print(f"    Applied: inv(pca_transform) only (no scale_mat)")
        else:
            print(f"    Applied: scale_mat only (no pca_transform)")
    else:
        # 方法2: 使用 Calibration 相机位置对齐（MILo方法，fallback）
        print(f"[DTU-EVAL] Loading DTU calibration cameras...")
        dtu_cameras = load_dtu_camera(dtu_path)
        if len(dtu_cameras) == 0:
            print(f"[DTU-EVAL] Error: No calibration cameras found at {dtu_path}/Calibration/cal18/")
            print(f"           and no transform.txt or cameras.npz found")
            return None
        print(f"    Loaded {len(dtu_cameras)} cameras")

        print(f"[DTU-EVAL] Aligning mesh to DTU world coordinates (camera-based)...")
        aligned_verts, scale, R, t = align_mesh_to_dtu(
            np.asarray(mesh.vertices), train_cameras, dtu_cameras
        )
        print(f"    Scale: {scale:.6f}")

    # 保存对齐后的mesh
    aligned_mesh = trimesh.Trimesh(vertices=aligned_verts, faces=mesh.faces, process=False)
    aligned_path = os.path.join(output_dir, "mesh_aligned.ply")
    aligned_mesh.export(aligned_path)
    print(f"    Saved: {aligned_path}")

    # 评估
    print(f"[DTU-EVAL] Evaluating (scan {scan_id})...")
    vis_dir = os.path.join(output_dir, "vis")
    results = eval_core(
        mesh_path=aligned_path,
        scan_id=scan_id,
        dataset_dir=dtu_path,
        vis_out_dir=vis_dir if write_visualizations else None,
        seed=seed,
        downsample_method=downsample_method,
        cpu_workers=cpu_workers,
    )

    # 保存结果
    import json
    if not write_visualizations:
        # Core None suppresses its small legacy JSON too; preserve that JSON.
        os.makedirs(vis_dir, exist_ok=True)
        with open(f'{vis_dir}/results.json', 'w') as fp:
            json.dump(results, fp, indent=2)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[DTU-EVAL] Results saved: {results_path}")

    return results
