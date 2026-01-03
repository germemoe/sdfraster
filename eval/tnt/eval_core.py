# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
# Copyright (c) 2017 Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, Vladlen Koltun
# ----------------------------------------------------------------------------
"""
TNT evaluation core functions.

Reference: MILo/milo/eval/tnt/evaluation.py
"""

import copy
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def read_alignment_transformation(filename: str) -> np.ndarray:
    """读取对齐变换矩阵"""
    with open(filename) as f:
        data = json.load(f)
    return np.asarray(data["transformation"]).reshape((4, 4)).transpose()


def write_color_distances(path: str, pcd, distances: np.ndarray, max_distance: float):
    """将距离值写入点云颜色"""
    cmap = plt.get_cmap("hot_r")
    distances = np.array(distances)
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def compute_fscore(precision: float, recall: float) -> float:
    """计算 F-score"""
    if precision + recall == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def evaluate_point_clouds(
    source_pcd,
    target_pcd,
    trans: np.ndarray,
    crop_volume,
    voxel_size: float,
    threshold: float,
    output_dir: str = None,
    scene_name: str = "scene",
    plot_stretch: float = 5.0,
    verbose: bool = True,
):
    """
    评估两个点云之间的距离。

    Args:
        source_pcd: 重建点云（Open3D PointCloud）
        target_pcd: GT 点云（Open3D PointCloud）
        trans: 对齐变换矩阵 (4x4)
        crop_volume: 裁剪体积（SelectionPolygonVolume）
        voxel_size: 下采样体素大小
        threshold: F-score 的距离阈值（tau）
        output_dir: 输出目录（可选，用于保存可视化）
        scene_name: 场景名称
        plot_stretch: 绘图范围倍数
        verbose: 是否输出详细信息

    Returns:
        dict: 包含 precision, recall, fscore 等结果
    """
    if verbose:
        print("[TNT-EVAL] Evaluating point clouds...")

    # 复制并变换源点云
    s = copy.deepcopy(source_pcd)
    s.transform(trans)

    # 裁剪点云
    if crop_volume is not None:
        s = crop_volume.crop_point_cloud(s)
        if verbose:
            print(f"    Source points after crop: {len(s.points)}")
    else:
        if verbose:
            print("    No bounding box provided, using full point cloud")

    # 下采样
    s = s.voxel_down_sample(voxel_size)
    s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

    # 处理目标点云
    t = copy.deepcopy(target_pcd)
    if crop_volume is not None:
        t = crop_volume.crop_point_cloud(t)
    t = t.voxel_down_sample(voxel_size)
    t.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

    if verbose:
        print(f"    Source: {len(s.points)} points, Target: {len(t.points)} points")

    # 计算双向 Chamfer 距离
    distance1 = np.asarray(s.compute_point_cloud_distance(t))  # source -> target (precision)
    distance2 = np.asarray(t.compute_point_cloud_distance(s))  # target -> source (recall)

    # 计算 precision, recall, F-score
    if len(distance1) > 0 and len(distance2) > 0:
        precision = float(np.sum(distance1 < threshold)) / float(len(distance1))
        recall = float(np.sum(distance2 < threshold)) / float(len(distance2))
        fscore = compute_fscore(precision, recall)

        # 计算累积分布（用于绘图）
        bins = np.arange(0, threshold * plot_stretch, threshold / 100)

        hist1, edges1 = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist1).astype(float) / len(distance1)

        hist2, edges2 = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist2).astype(float) / len(distance2)
    else:
        precision = 0.0
        recall = 0.0
        fscore = 0.0
        edges1 = np.array([0])
        cum_source = np.array([0])
        edges2 = np.array([0])
        cum_target = np.array([0])

    # 保存可视化结果
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存带颜色的点云
        precision_path = os.path.join(output_dir, f"{scene_name}.precision.ply")
        recall_path = os.path.join(output_dir, f"{scene_name}.recall.ply")
        write_color_distances(precision_path, s, distance1, 3 * threshold)
        write_color_distances(recall_path, t, distance2, 3 * threshold)

        # 保存结果
        np.savetxt(os.path.join(output_dir, f"{scene_name}.recall.txt"), cum_target)
        np.savetxt(os.path.join(output_dir, f"{scene_name}.precision.txt"), cum_source)
        np.savetxt(
            os.path.join(output_dir, f"{scene_name}.prf_tau_plotstr.txt"),
            np.array([precision, recall, fscore, threshold, plot_stretch])
        )

    results = {
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "threshold": threshold,
        "num_source_points": len(s.points),
        "num_target_points": len(t.points),
        "mean_d2s": float(np.mean(distance1)) if len(distance1) > 0 else 0.0,
        "mean_s2d": float(np.mean(distance2)) if len(distance2) > 0 else 0.0,
        "edges_source": edges1,
        "cum_source": cum_source,
        "edges_target": edges2,
        "cum_target": cum_target,
    }

    if verbose:
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F-score: {fscore:.4f}")

    return results
