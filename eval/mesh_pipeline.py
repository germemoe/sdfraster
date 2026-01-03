"""
Mesh Processing Pipeline

统一的mesh提取、处理和评估流水线。
可以从train.py调用，也可以作为独立脚本运行。

支持的提取方法：
- mt: 标准Marching Tetrahedra
- mt_milo: MILo风格MT处理
- tsdf: TSDF Fusion

支持的评估数据集：
- dtu: DTU数据集评估
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class MeshPipeline:
    """
    统一的Mesh处理流水线。

    使用方法：
        pipeline = MeshPipeline(model, train_cameras, scene_info, output_dir)

        # 提取mesh
        verts, faces = pipeline.extract(method='mt_milo')

        # 评估（DTU）
        results = pipeline.evaluate_dtu(verts, faces, dtu_path, scan_id)
    """

    def __init__(
        self,
        model,
        train_cameras: List,
        scene_info,
        output_dir: str,
    ):
        """
        初始化流水线。

        Args:
            model: 训练好的模型（需要有vertices, indices, query_vert_sdf等属性）
            train_cameras: 训练相机列表
            scene_info: 场景信息（包含transform等）
            output_dir: 输出目录
        """
        self.model = model
        self.train_cameras = train_cameras
        self.scene_info = scene_info
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 提取方法 ==========

    def extract_mt(
        self,
        sdf_threshold: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用标准Marching Tetrahedra提取mesh。

        Returns:
            verts: (N, 3) numpy array
            faces: (M, 3) numpy array
        """
        from utils.tetmesh import marching_tetrahedra

        with torch.no_grad():
            vert_sdf = self.model.query_vert_sdf() - sdf_threshold
            verts, faces = marching_tetrahedra(
                self.model.vertices,
                self.model.indices,
                vert_sdf,
                num_internal=self.model.num_int_verts
            )

        return verts.cpu().numpy(), faces.cpu().numpy()

    def extract_mt_milo(
        self,
        sdf_threshold: float = 0.0,
        filter_large_edges: bool = True,
        collapse_large_edges: bool = True,
        remove_oof: bool = True,
        edge_scale: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用MILo风格处理的MT提取mesh。

        Returns:
            verts: (N, 3) numpy array
            faces: (M, 3) numpy array
        """
        from utils.tetmesh import marching_tetrahedra_with_interpolation
        from utils.mesh_process_milo import process_mt_mesh_milo

        with torch.no_grad():
            vert_sdf = self.model.query_vert_sdf() - sdf_threshold

            edge_endpoints, edge_sdf, faces, edge_vertex_ids = marching_tetrahedra_with_interpolation(
                self.model.vertices,
                self.model.indices,
                vert_sdf,
                num_internal=self.model.num_int_verts
            )

            if faces.shape[0] == 0:
                print("[MT-MILo] No faces extracted")
                return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

            verts, faces, _, _ = process_mt_mesh_milo(
                edge_endpoints=edge_endpoints,
                edge_sdf=edge_sdf,
                faces=faces,
                edge_vertex_ids=edge_vertex_ids,
                cameras=self.train_cameras,
                filter_large_edges=filter_large_edges,
                collapse_large_edges=collapse_large_edges,
                remove_oof_vertices=remove_oof,
                edge_scale_threshold=edge_scale,
                verbose=True
            )

        return verts.cpu().numpy(), faces.cpu().numpy()

    def extract_tsdf(
        self,
        voxel_size: float = 0.002,
        depth_trunc: float = 8.0,
        depth_ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用TSDF Fusion提取mesh。

        Returns:
            verts: (N, 3) numpy array
            faces: (M, 3) numpy array
        """
        from scripts.extract_mesh_tsdf import TSDFExtractor

        extractor = TSDFExtractor(
            voxel_size=voxel_size,
            depth_trunc=depth_trunc,
            depth_ratio=depth_ratio,
        )

        # 渲染深度图
        extractor.reconstruction(self.train_cameras, self.model)

        # 提取mesh
        mesh = extractor.extract_mesh()

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        return verts, faces

    def extract(
        self,
        method: str = 'mt',
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        统一的提取接口。

        Args:
            method: 'mt', 'mt_milo', 'tsdf'
            **kwargs: 传递给具体方法的参数

        Returns:
            verts, faces
        """
        if method == 'mt':
            return self.extract_mt(**kwargs)
        elif method == 'mt_milo':
            return self.extract_mt_milo(**kwargs)
        elif method == 'tsdf':
            return self.extract_tsdf(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    # ========== 后处理 ==========

    def post_process(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        num_cluster: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        后处理：移除小的disconnected clusters。
        """
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        components = mesh.split(only_watertight=False)
        if len(components) == 0:
            return verts, faces

        components = sorted(components, key=lambda x: len(x.faces), reverse=True)
        components = components[:num_cluster]

        if len(components) == 1:
            result = components[0]
        else:
            result = trimesh.util.concatenate(components)

        return np.asarray(result.vertices), np.asarray(result.faces)

    def cull_mesh(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        dilation_radius: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用相机mask过滤mesh顶点（MILo方法）。
        """
        from eval.dtu.evaluate import cull_mesh as cull_mesh_func

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh = cull_mesh_func(self.train_cameras, mesh, dilation_radius=dilation_radius)

        return np.asarray(mesh.vertices), np.asarray(mesh.faces)

    # ========== 坐标变换 ==========

    def transform_to_world(
        self,
        verts: np.ndarray,
        apply_scale_mat: bool = True,
    ) -> np.ndarray:
        """
        将mesh从训练坐标系变换到世界坐标系。

        对于DTU数据集：
        1. 应用inv_transform（scene_info.transform的逆）
        2. 应用scale_mat（如果cameras.npz存在）

        Args:
            verts: (N, 3) 训练坐标系下的顶点
            apply_scale_mat: 是否应用scale_mat（DTU需要）

        Returns:
            verts_world: (N, 3) 世界坐标系下的顶点
        """
        # 1. 逆PCA变换
        inv_transform = np.linalg.inv(self.scene_info.transform)
        verts_hom = np.hstack([verts, np.ones((verts.shape[0], 1))])
        verts_colmap = (inv_transform @ verts_hom.T).T[:, :3]

        # 2. 应用scale_mat（如果存在）
        if apply_scale_mat:
            cameras_npz = self.scene_info.path / 'cameras.npz' if hasattr(self.scene_info, 'path') else None
            if cameras_npz is None:
                # 尝试从其他地方获取路径
                cameras_npz = Path(str(self.output_dir).replace('/output/', '/dataset/')) / 'cameras.npz'

            if cameras_npz and cameras_npz.exists():
                camera_dict = np.load(str(cameras_npz))
                scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
                verts_hom = np.hstack([verts_colmap, np.ones((verts_colmap.shape[0], 1))])
                verts_world = (scale_mat @ verts_hom.T).T[:, :3]
            else:
                verts_world = verts_colmap
        else:
            verts_world = verts_colmap

        return verts_world

    # ========== DTU评估 ==========

    def evaluate_dtu(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        dtu_path: str,
        scan_id: int,
        cull: bool = True,
        dilation_radius: int = 6,
        output_subdir: str = 'eval_dtu',
    ) -> Optional[Dict[str, float]]:
        """
        完整的DTU mesh评估流程。

        流程（复用MILo方法）：
        1. cull_mesh（使用相机mask，在训练坐标系下）
        2. 通过相机位置对齐到DTU世界坐标系
        3. 调用eval_core计算Chamfer距离

        Args:
            verts: (N, 3) 训练坐标系下的顶点
            faces: (M, 3) 面片索引
            dtu_path: DTU数据集根目录
            scan_id: DTU scan ID
            cull: 是否进行cull_mesh
            dilation_radius: mask膨胀半径
            output_subdir: 输出子目录名

        Returns:
            评估结果字典，包含 mean_d2s, mean_s2d, overall
        """
        from eval.dtu.evaluate import (
            cull_mesh as cull_mesh_func,
            load_dtu_cameras,
            align_mesh_to_dtu,
        )
        from eval.dtu.eval_core import evaluate_mesh as eval_core

        output_dir = self.output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        print(f"[DTU-EVAL] Input mesh: {len(verts)} vertices, {len(faces)} faces")

        # 1. Cull mesh（训练坐标系下）
        if cull:
            print(f"[DTU-EVAL] Culling mesh with camera masks (dilation={dilation_radius})...")
            mesh = cull_mesh_func(self.train_cameras, mesh, dilation_radius=dilation_radius)
            culled_path = output_dir / "mesh_culled.ply"
            mesh.export(str(culled_path))
            print(f"    Saved: {culled_path}")
            print(f"    After cull: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # 2. 加载DTU相机并对齐
        print(f"[DTU-EVAL] Loading DTU calibration cameras from {dtu_path}...")
        dtu_cameras = load_dtu_cameras(dtu_path)
        if len(dtu_cameras) == 0:
            print(f"[DTU-EVAL] Error: No DTU calibration cameras found")
            print(f"    Expected: {dtu_path}/Calibration/cal18/pos_*.txt")
            return None
        print(f"    Loaded {len(dtu_cameras)} DTU cameras")

        print(f"[DTU-EVAL] Aligning mesh to DTU world coordinates...")
        mesh, scale, R, t = align_mesh_to_dtu(mesh, self.train_cameras, dtu_cameras)
        aligned_path = output_dir / "mesh_aligned.ply"
        mesh.export(str(aligned_path))
        print(f"    Saved: {aligned_path}")
        print(f"    Scale factor: {scale:.6f}")

        # 3. 评估
        print(f"[DTU-EVAL] Evaluating mesh (scan {scan_id})...")
        vis_dir = output_dir / "vis"
        results = eval_core(
            mesh_path=str(aligned_path),
            scan_id=scan_id,
            dataset_dir=dtu_path,
            vis_out_dir=str(vis_dir),
        )

        # 保存结果
        import json
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[DTU-EVAL] Results saved: {results_path}")

        return results

    # ========== 保存 ==========

    def save_mesh(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        name: str,
        compute_normals: bool = True,
    ) -> str:
        """
        保存mesh到文件。

        Args:
            verts: (N, 3) 顶点
            faces: (M, 3) 面片
            name: 文件名（不含扩展名）
            compute_normals: 是否计算法线

        Returns:
            保存的文件路径
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        if compute_normals:
            mesh.compute_vertex_normals()

        path = self.output_dir / f"{name}.ply"
        o3d.io.write_triangle_mesh(str(path), mesh)
        print(f"[MESH] Saved: {path} ({len(verts)} verts, {len(faces)} faces)")

        return str(path)


def run_full_pipeline(
    model,
    train_cameras: List,
    scene_info,
    output_dir: str,
    methods: List[str] = ['mt', 'mt_milo', 'tsdf'],
    eval_dtu: bool = True,
    dtu_path: str = None,
    scan_id: int = None,
    cull: bool = True,
    post_process: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    运行完整的mesh提取和评估流水线。

    Args:
        model: 模型
        train_cameras: 训练相机
        scene_info: 场景信息
        output_dir: 输出目录
        methods: 要运行的提取方法列表
        eval_dtu: 是否进行DTU评估
        dtu_path: DTU数据集路径
        scan_id: DTU scan ID
        cull: 是否进行cull_mesh
        post_process: 是否进行后处理

    Returns:
        所有方法的评估结果
    """
    pipeline = MeshPipeline(model, train_cameras, scene_info, output_dir)

    all_results = {}

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"[PIPELINE] Method: {method}")
        print('=' * 60)

        # 提取
        verts, faces = pipeline.extract(method=method, **kwargs.get(method, {}))
        print(f"[PIPELINE] Extracted: {len(verts)} vertices, {len(faces)} faces")

        if len(faces) == 0:
            print(f"[PIPELINE] Warning: No faces extracted for {method}")
            continue

        # 后处理
        if post_process:
            verts, faces = pipeline.post_process(verts, faces)
            print(f"[PIPELINE] After post-process: {len(verts)} vertices, {len(faces)} faces")

        # 保存
        pipeline.save_mesh(verts, faces, f"mesh_{method}")

        # DTU评估
        if eval_dtu and dtu_path and scan_id:
            results = pipeline.evaluate_dtu(
                verts, faces,
                dtu_path=dtu_path,
                scan_id=scan_id,
                cull=cull,
                output_subdir=f'eval_dtu_{method}'
            )
            all_results[method] = results

    return all_results
