# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
# Copyright (c) 2017 Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, Vladlen Koltun
# ----------------------------------------------------------------------------
"""
TNT trajectory alignment and registration functions.

Reference: MILo/milo/eval/tnt/registration.py
"""

import copy
import numpy as np
import open3d as o3d


MAX_POINT_NUMBER = 4e6


class CameraPose:
    """相机位姿类"""
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return f"Metadata: {self.metadata}\nPose:\n{np.array_str(self.pose)}"


def convert_trajectory_to_pointcloud(traj: list):
    """将轨迹转换为点云（相机位置）"""
    pcd = o3d.geometry.PointCloud()
    for t in traj:
        pcd.points.append(t.pose[:3, 3])
    return pcd


def read_trajectory(filename: str) -> list:
    """从 log 文件读取轨迹"""
    traj = []
    with open(filename, "r") as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=" \t")
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def read_mapping(filename: str):
    """读取帧映射文件"""
    mapping = []
    with open(filename, "r") as f:
        n_sampled_frames = int(f.readline())
        n_total_frames = int(f.readline())
        mapping = np.zeros(shape=(n_sampled_frames, 2))
        metastr = f.readline()
        for i in range(n_sampled_frames):
            metadata = list(map(int, metastr.split()))
            mapping[i, :] = metadata
            metastr = f.readline()
    return [n_sampled_frames, n_total_frames, mapping]


def gen_sparse_trajectory(mapping, f_trajectory):
    """根据映射生成稀疏轨迹"""
    sparse_traj = []
    for m in mapping:
        sparse_traj.append(f_trajectory[int(m[1] - 1)])
    return sparse_traj


def trajectory_alignment(map_file, traj_to_register: list, gt_traj_col: list,
                         gt_trans: np.ndarray, scene: str) -> np.ndarray:
    """
    轨迹对齐：使用 RANSAC 将重建轨迹与 GT 轨迹对齐。

    Args:
        map_file: 帧映射文件路径（可选）
        traj_to_register: 要对齐的轨迹（CameraPose 列表）
        gt_traj_col: GT COLMAP 轨迹（CameraPose 列表）
        gt_trans: GT 变换矩阵
        scene: 场景名称

    Returns:
        对齐变换矩阵 (4x4)
    """
    # 将 GT 轨迹转换为点云并应用 GT 变换
    traj_pcd_col = convert_trajectory_to_pointcloud(gt_traj_col)
    if gt_trans is not None:
        traj_pcd_col.transform(gt_trans)

    # 创建对应关系
    corres = o3d.utility.Vector2iVector(
        np.asarray(list(map(lambda x: [x, x], range(len(gt_traj_col)))))
    )

    # RANSAC 参数
    rr = o3d.pipelines.registration.RANSACConvergenceCriteria()
    rr.max_iteration = 100000

    # 处理长轨迹（稀疏化）
    if len(traj_to_register) > 1600 and map_file is not None:
        n_sampled_frames, n_total_frames, mapping = read_mapping(map_file)
        traj_col2 = gen_sparse_trajectory(mapping, traj_to_register)
        traj_to_register_pcd = convert_trajectory_to_pointcloud(traj_col2)
    else:
        traj_to_register_pcd = convert_trajectory_to_pointcloud(traj_to_register)

    # RANSAC 对齐
    reg = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        traj_to_register_pcd,
        traj_pcd_col,
        corres,
        0.2,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        6,
        criteria=rr,
    )

    return reg.transformation


def crop_and_downsample(
    pcd,
    crop_volume,
    down_sample_method: str = "voxel",
    voxel_size: float = 0.01,
    trans: np.ndarray = None,
):
    """裁剪并下采样点云"""
    if trans is None:
        trans = np.identity(4)

    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(trans)
    pcd_crop = crop_volume.crop_point_cloud(pcd_copy)

    if down_sample_method == "voxel":
        return pcd_crop.voxel_down_sample(voxel_size)
    elif down_sample_method == "uniform":
        n_points = len(pcd_crop.points)
        if n_points > MAX_POINT_NUMBER:
            ds_rate = int(round(n_points / float(MAX_POINT_NUMBER)))
            return pcd_crop.uniform_down_sample(ds_rate)
    return pcd_crop


def registration_icp(
    source,
    gt_target,
    init_trans: np.ndarray,
    crop_volume,
    threshold: float,
    max_itr: int,
    down_sample_method: str = "voxel",
    voxel_size: float = 0.01,
    verbose: bool = True,
):
    """
    ICP 精细配准。

    Args:
        source: 源点云
        gt_target: 目标点云（GT）
        init_trans: 初始变换
        crop_volume: 裁剪体积
        threshold: ICP 阈值
        max_itr: 最大迭代次数
        down_sample_method: 下采样方法 ("voxel" 或 "uniform")
        voxel_size: 体素大小
        verbose: 是否输出详细信息

    Returns:
        配准结果（包含 transformation）
    """
    if verbose:
        print(f"[Registration] threshold: {threshold:.4f}, voxel: {voxel_size:.4f}")

    s = crop_and_downsample(
        source, crop_volume, down_sample_method, voxel_size, init_trans
    )
    t = crop_and_downsample(
        gt_target, crop_volume, down_sample_method, voxel_size
    )

    reg = o3d.pipelines.registration.registration_icp(
        s, t, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, max_iteration=max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)

    return reg


def refine_registration(
    source_pcd,
    gt_pcd,
    init_trans: np.ndarray,
    crop_volume,
    tau: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    多阶段 ICP 精细配准。

    Reference: MILo TNT evaluation 的三阶段配准。

    Args:
        source_pcd: 源点云
        gt_pcd: GT 点云
        init_trans: 初始变换
        crop_volume: 裁剪体积
        tau: 场景 tau 值
        verbose: 是否输出详细信息

    Returns:
        最终变换矩阵 (4x4)
    """
    if verbose:
        print("[TNT-EVAL] Refining registration...")

    # Stage 1: 粗配准
    r1 = registration_icp(
        source_pcd, gt_pcd, init_trans, crop_volume,
        threshold=tau * 80, max_itr=20,
        down_sample_method="voxel", voxel_size=tau,
        verbose=verbose
    )

    # Stage 2: 中等精度配准
    r2 = registration_icp(
        source_pcd, gt_pcd, r1.transformation, crop_volume,
        threshold=tau * 20, max_itr=20,
        down_sample_method="voxel", voxel_size=tau / 2.0,
        verbose=verbose
    )

    # Stage 3: 精细配准
    r3 = registration_icp(
        source_pcd, gt_pcd, r2.transformation, crop_volume,
        threshold=2 * tau, max_itr=20,
        down_sample_method="uniform",
        verbose=verbose
    )

    return r3.transformation
