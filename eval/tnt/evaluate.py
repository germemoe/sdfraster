"""
TNT (Tanks and Temples) Mesh Evaluation

完整的 TNT mesh 评估流程，复用 MILo 的评估代码。

评估流程：
1. 加载 mesh 并转换为点云
2. 加载 GT 点云
3. 轨迹对齐（RANSAC + ICP）
4. 裁剪到评估区域
5. 计算 F-score (precision, recall)

坐标对齐方法：
- TNT 使用轨迹对齐，将重建的相机轨迹与 GT COLMAP 轨迹对齐
- 然后通过 ICP 精细配准

Reference: MILo/milo/eval/tnt/run.py
"""

import os
import json
import numpy as np
import open3d as o3d
import trimesh
from os import path

from .config import get_scene_tau, SCENES_TAU_DICT
from .registration import (
    CameraPose,
    read_trajectory,
    trajectory_alignment,
    refine_registration,
)
from .eval_core import evaluate_point_clouds


def load_pca_transform(model_output_path: str) -> np.ndarray:
    """
    从模型输出目录加载 PCA 变换矩阵。

    Args:
        model_output_path: 模型输出目录（包含 transform.txt）

    Returns:
        pca_transform: (4, 4) numpy array, 或 None 如果不存在
    """
    transform_path = path.join(model_output_path, 'transform.txt')
    if not os.path.exists(transform_path):
        return None
    return np.loadtxt(transform_path).astype(np.float64)


def mesh_to_pointcloud(mesh_path: str) -> o3d.geometry.PointCloud:
    """
    从 mesh 创建点云（包括顶点和面中心点）。

    Reference: MILo TNT evaluation 中的点云采样方法。
    """
    mesh = trimesh.load_mesh(mesh_path)

    # 添加面中心点以增加采样密度
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)
    vertices = np.concatenate([mesh.vertices, face_centers], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    return pcd


def cameras_to_trajectory(train_cameras: list, pca_transform: np.ndarray = None) -> list:
    """
    将训练相机转换为 CameraPose 轨迹列表。

    Args:
        train_cameras: 训练相机列表
        pca_transform: PCA 变换矩阵（如果有）

    Returns:
        CameraPose 列表（已逆变换到归一化坐标）
    """
    trajectory = []

    # PCA 逆变换
    pca_inv = None
    if pca_transform is not None:
        pca_inv = np.linalg.inv(pca_transform)

    for cam in train_cameras:
        # 获取 c2w 矩阵
        c2w = (cam.world_view_transform.T).inverse()
        c2w = c2w.cpu().numpy()

        # 转换为 4x4
        if c2w.shape[0] == 3:
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :] = c2w
            c2w = c2w_4x4

        # 应用 PCA 逆变换（训练坐标 -> 归一化坐标）
        if pca_inv is not None:
            c2w = pca_inv @ c2w

        trajectory.append(CameraPose(meta=None, mat=c2w))

    return trajectory


def align_mesh_to_tnt(
    mesh_vertices: np.ndarray,
    pca_transform: np.ndarray = None,
) -> np.ndarray:
    """
    将 mesh 从训练坐标系变换到归一化坐标系。

    TNT 的对齐是通过轨迹配准完成的，这里只做 PCA 逆变换。

    Args:
        mesh_vertices: (N, 3) 训练坐标系下的顶点
        pca_transform: PCA 变换矩阵

    Returns:
        aligned_vertices: (N, 3) 归一化坐标系下的顶点
    """
    N = mesh_vertices.shape[0]
    vertices = mesh_vertices.copy()

    if pca_transform is not None:
        pca_inv = np.linalg.inv(pca_transform)
        vertices_hom = np.hstack([vertices, np.ones((N, 1))])
        vertices = (pca_inv @ vertices_hom.T).T[:, :3]

    return vertices


def evaluate_tnt_mesh(
    mesh_path: str,
    train_cameras: list,
    tnt_path: str,
    scene_name: str,
    output_dir: str = None,
    model_output_path: str = None,
    cull: bool = False,
    min_visible_views: int = 20,
):
    """
    完整的 TNT mesh 评估流程。

    Args:
        mesh_path: 输入 mesh 路径（训练坐标系）
        train_cameras: 训练相机列表
        tnt_path: TNT 场景目录（包含 {scene}.ply, {scene}.json, {scene}_trans.txt 等）
        scene_name: 场景名称（如 "Barn", "Truck"）
        output_dir: 输出目录
        model_output_path: 模型输出目录（包含 transform.txt）
        cull: 是否进行 mesh 裁剪
        min_visible_views: mesh 裁剪的最少可见视图数

    Returns:
        评估结果字典
    """
    if output_dir is None:
        output_dir = os.path.dirname(mesh_path)
    os.makedirs(output_dir, exist_ok=True)

    # 获取场景 tau
    if scene_name not in SCENES_TAU_DICT:
        print(f"[TNT-EVAL] Warning: Unknown scene '{scene_name}', using default tau=0.01")
        tau = 0.01
    else:
        tau = get_scene_tau(scene_name)

    print(f"\n{'='*60}")
    print(f"[TNT-EVAL] Evaluating {scene_name}")
    print(f"{'='*60}")
    print(f"    Tau (threshold): {tau}")

    # 加载 PCA 变换
    pca_transform = None
    if model_output_path is not None:
        pca_transform = load_pca_transform(model_output_path)
        if pca_transform is not None:
            print(f"    Loaded PCA transform from {model_output_path}/transform.txt")

    # 加载并处理 mesh
    print(f"[TNT-EVAL] Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    print(f"    Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # Mesh 裁剪（可选）
    if cull:
        from .cull_mesh import cull_mesh_with_cameras
        print(f"[TNT-EVAL] Culling mesh (min_visible_views={min_visible_views})...")
        mesh = cull_mesh_with_cameras(mesh, train_cameras, min_visible_views)
        culled_path = os.path.join(output_dir, "mesh_culled.ply")
        mesh.export(culled_path)
        print(f"    Saved: {culled_path}")

    # 应用 PCA 逆变换（训练坐标 -> 归一化坐标）
    if pca_transform is not None:
        print("[TNT-EVAL] Applying PCA inverse transform...")
        mesh.vertices = align_mesh_to_tnt(np.asarray(mesh.vertices), pca_transform)

    # 转换 mesh 为点云
    print("[TNT-EVAL] Converting mesh to point cloud...")
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)
    vertices = np.concatenate([mesh.vertices, face_centers], axis=0)
    recon_pcd = o3d.geometry.PointCloud()
    recon_pcd.points = o3d.utility.Vector3dVector(vertices)
    print(f"    Point cloud: {len(recon_pcd.points)} points")

    # 加载 GT 点云
    gt_ply_path = os.path.join(tnt_path, f"{scene_name}.ply")
    if not os.path.exists(gt_ply_path):
        print(f"[TNT-EVAL] Error: GT point cloud not found at {gt_ply_path}")
        return None
    print(f"[TNT-EVAL] Loading GT point cloud: {gt_ply_path}")
    gt_pcd = o3d.io.read_point_cloud(gt_ply_path)
    print(f"    GT points: {len(gt_pcd.points)}")

    # 加载 GT 变换
    trans_path = os.path.join(tnt_path, f"{scene_name}_trans.txt")
    if os.path.exists(trans_path):
        gt_trans = np.loadtxt(trans_path)
        print(f"    Loaded GT transform from {trans_path}")
    else:
        gt_trans = None
        print(f"    No GT transform found")

    # 加载 COLMAP SfM 轨迹
    colmap_log_path = os.path.join(tnt_path, f"{scene_name}_COLMAP_SfM.log")
    if not os.path.exists(colmap_log_path):
        print(f"[TNT-EVAL] Error: COLMAP SfM log not found at {colmap_log_path}")
        return None
    print(f"[TNT-EVAL] Loading COLMAP trajectory: {colmap_log_path}")
    gt_traj = read_trajectory(colmap_log_path)
    print(f"    GT trajectory: {len(gt_traj)} poses")

    # 转换训练相机为轨迹
    print("[TNT-EVAL] Converting train cameras to trajectory...")
    train_traj = cameras_to_trajectory(train_cameras, pca_transform)
    print(f"    Train trajectory: {len(train_traj)} poses")

    # 轨迹对齐
    print("[TNT-EVAL] Aligning trajectories (RANSAC)...")
    init_trans = trajectory_alignment(None, train_traj, gt_traj, gt_trans, scene_name)
    print(f"    Initial alignment computed")

    # 加载裁剪区域
    crop_json_path = os.path.join(tnt_path, f"{scene_name}.json")
    if os.path.exists(crop_json_path):
        crop_volume = o3d.visualization.read_selection_polygon_volume(crop_json_path)
        print(f"    Loaded crop volume from {crop_json_path}")
    else:
        crop_volume = None
        print(f"    No crop volume found")

    # ICP 精细配准
    print("[TNT-EVAL] Refining registration (ICP)...")
    final_trans = refine_registration(recon_pcd, gt_pcd, init_trans, crop_volume, tau)

    # 保存对齐后的点云
    aligned_pcd = o3d.geometry.PointCloud(recon_pcd)
    aligned_pcd.transform(final_trans)
    aligned_path = os.path.join(output_dir, "pcd_aligned.ply")
    o3d.io.write_point_cloud(aligned_path, aligned_pcd)
    print(f"    Saved aligned point cloud: {aligned_path}")

    # 评估
    print("[TNT-EVAL] Computing metrics...")
    results = evaluate_point_clouds(
        source_pcd=recon_pcd,
        target_pcd=gt_pcd,
        trans=final_trans,
        crop_volume=crop_volume,
        voxel_size=tau / 2.0,
        threshold=tau,
        output_dir=output_dir,
        scene_name=scene_name,
    )

    # 保存结果
    results_to_save = {
        "scene": scene_name,
        "precision": results["precision"],
        "recall": results["recall"],
        "fscore": results["fscore"],
        "threshold": tau,
        "num_source_points": results["num_source_points"],
        "num_target_points": results["num_target_points"],
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"[TNT-EVAL] Results saved: {results_path}")

    print(f"\n{'='*60}")
    print(f"[TNT-EVAL] Results for {scene_name}")
    print(f"{'='*60}")
    print(f"    Threshold (tau): {tau}")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall: {results['recall']:.4f}")
    print(f"    F-score: {results['fscore']:.4f}")
    print(f"{'='*60}\n")

    return results


def get_tnt_info(dataset_path: str):
    """
    从数据集路径提取 TNT 信息。

    Args:
        dataset_path: 数据集路径

    Returns:
        (tnt_root, scene_name, is_tnt)
    """
    src_path = os.path.normpath(str(dataset_path))
    scene_folder = os.path.basename(src_path)
    tnt_root = os.path.dirname(src_path)

    # 检查是否为 TNT 数据集
    # TNT 场景名通常是 Barn, Truck 等，数据目录下应有 {scene}.ply, {scene}.json 等文件
    scene_name = scene_folder

    # 检查必要文件
    gt_ply = os.path.join(tnt_root, f"{scene_name}.ply")
    colmap_log = os.path.join(tnt_root, f"{scene_name}_COLMAP_SfM.log")

    is_tnt = os.path.exists(gt_ply) or os.path.exists(colmap_log) or scene_name in SCENES_TAU_DICT

    return tnt_root, scene_name, is_tnt
