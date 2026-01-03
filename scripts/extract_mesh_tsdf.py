#!/usr/bin/env python3
"""
TSDF Fusion Mesh Extraction Script (参考 MILo)

从训练好的模型中提取深度图，使用 TSDF 融合生成网格，并可选进行 DTU 评估。

用法:
    python scripts/extract_mesh_tsdf.py --model_path output/xxx --eval_dtu
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import open3d as o3d
import math

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def to_cam_open3d(viewpoint_stack):
    """Convert cameras to Open3D format."""
    camera_traj = []
    for viewpoint_cam in viewpoint_stack:
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height

        # 从 FOV 计算内参
        fx = W / (2 * math.tan(viewpoint_cam.fovx / 2.))
        fy = H / (2 * math.tan(viewpoint_cam.fovy / 2.))
        cx = W / 2.0
        cy = H / 2.0

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=W,
            height=H,
            cx=cx,
            cy=cy,
            fx=fx,
            fy=fy
        )

        # world_view_transform 是 W2V 的转置，需要转置回来得到 W2V
        extrinsic = viewpoint_cam.world_view_transform.T.cpu().numpy()

        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


def post_process_mesh(mesh, cluster_to_keep=50):
    """Post-process mesh to remove small disconnected components."""
    import copy
    print(f"Post processing mesh, keeping top {cluster_to_keep} clusters...")
    mesh_0 = copy.deepcopy(mesh)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, _ = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep] if len(cluster_n_triangles) >= cluster_to_keep else 0
    n_cluster = max(n_cluster, 50)  # 过滤小于 50 个三角形的 cluster

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()

    print(f"  Vertices: {len(mesh.vertices)} -> {len(mesh_0.vertices)}")
    print(f"  Faces: {len(mesh.triangles)} -> {len(mesh_0.triangles)}")

    return mesh_0


class TSDFExtractor:
    """TSDF Fusion based mesh extractor."""

    def __init__(self, model, depth_ratio=0.6, tile_size=4, min_t=0.4):
        self.model = model
        self.depth_ratio = depth_ratio
        self.tile_size = tile_size
        self.min_t = min_t
        self.depthmaps = []
        self.rgbmaps = []
        self.viewpoint_stack = []
        self.radius = None
        self.center = None

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """Render depth and RGB for all cameras."""
        self.depthmaps = []
        self.rgbmaps = []
        self.viewpoint_stack = viewpoint_stack

        device = next(self.model.parameters()).device

        for viewpoint_cam in tqdm(viewpoint_stack, desc="Rendering views"):
            render_pkg = render(
                viewpoint_cam, self.model,
                scene_scaling=self.model.scene_scaling,
                min_t=self.min_t,
                tile_size=self.tile_size
            )

            rgb = render_pkg['render']  # (3, H, W)
            expected_depth = render_pkg['depth']  # (H, W)
            median_depth = render_pkg['depth_med']  # (H, W)

            # MILo 风格：混合期望深度和中位深度
            surface_depth = self.depth_ratio * expected_depth + (1.0 - self.depth_ratio) * median_depth

            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(surface_depth.cpu())

        self._estimate_bounding_sphere()

    def _estimate_bounding_sphere(self):
        """Estimate bounding sphere from camera poses."""
        torch.cuda.empty_cache()

        c2ws = []
        for cam in self.viewpoint_stack:
            w2v = cam.world_view_transform.T.cpu().numpy()
            c2w = np.linalg.inv(w2v)
            c2ws.append(c2w)
        c2ws = np.array(c2ws)

        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = focus_point_fn(poses)

        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()

        print(f"Estimated bounding radius: {self.radius:.2f}")

    @torch.no_grad()
    def extract_mesh(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3.0, mask_background=True):
        """Perform TSDF fusion and extract mesh."""
        print("Running TSDF volume integration...")
        print(f"  voxel_size: {voxel_size}")
        print(f"  sdf_trunc: {sdf_trunc}")
        print(f"  depth_trunc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        cam_o3d_list = to_cam_open3d(self.viewpoint_stack)

        for i, cam_o3d in tqdm(enumerate(cam_o3d_list), desc="TSDF integration", total=len(cam_o3d_list)):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # 使用 mask（如果有）
            if mask_background:
                gt_mask = getattr(self.viewpoint_stack[i], 'gt_alpha_mask', None)
                if gt_mask is not None:
                    mask = gt_mask.cpu().squeeze()
                    if mask.dim() == 2:
                        depth = depth.clone()
                        depth[mask < 0.5] = 0

            # 创建 Open3D RGBD 图像
            rgb_np = np.asarray(
                np.clip(rgb.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255,
                order="C", dtype=np.uint8
            )
            depth_np = np.asarray(
                depth.unsqueeze(-1).numpy() if depth.dim() == 2 else depth.permute(1, 2, 0).numpy(),
                order="C"
            )
            if depth_np.ndim == 3:
                depth_np = depth_np[:, :, 0]

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_np),
                o3d.geometry.Image(depth_np.astype(np.float32)),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh


def main():
    parser = ArgumentParser(description="TSDF Fusion Mesh Extraction (MILo DTU style)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--depth_ratio", type=float, default=0.0, help="Blend ratio: 0.0=median only (MILo DTU default), 1.0=expected only")
    parser.add_argument("--voxel_size", type=float, default=0.002, help="Voxel size (MILo DTU: 0.002)")
    parser.add_argument("--depth_trunc", type=float, default=8.0, help="Depth truncation (MILo DTU: 8.0)")
    parser.add_argument("--sdf_trunc", type=float, default=0.01, help="SDF truncation (MILo DTU: ~5*voxel)")
    parser.add_argument("--num_cluster", type=int, default=50, help="Number of clusters to keep in post-processing")
    parser.add_argument("--eval_dtu", action="store_true", help="Run DTU evaluation")
    parser.add_argument("--skip_post_process", action="store_true", help="Skip post-processing")
    parser.add_argument("--cull_mesh", action="store_true", default=True, help="Cull mesh using camera masks (MILo style)")
    parser.add_argument("--no_cull", dest="cull_mesh", action="store_false", help="Skip mesh culling")
    parser.add_argument("--cull_dilation", type=int, default=6, help="Dilation radius for mask (MILo default: 6)")
    args = parser.parse_args()

    model_path = Path(args.model_path)

    # 加载配置
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {model_path}")
        return

    config = Args.load_from_json(str(config_path))

    # 加载数据集
    print(f"Loading dataset from {config.dataset_path}...")
    train_cameras, test_cameras, scene_info = loader.load_dataset(
        config.dataset_path, config.image_folder,
        data_device='cpu', eval=True, resolution=config.resolution
    )

    # 加载模型
    print(f"Loading model from {model_path}...")
    device = torch.device('cuda')
    model = Model.load_ckpt(model_path, device)
    model.eval()

    # TSDF 提取
    extractor = TSDFExtractor(
        model,
        depth_ratio=args.depth_ratio,
        tile_size=config.tile_size,
        min_t=getattr(config, 'min_t', 0.4)
    )

    print("Reconstructing from training views...")
    extractor.reconstruction(train_cameras)

    # 使用 MILo DTU 风格的固定参数
    voxel_size = args.voxel_size
    depth_trunc = args.depth_trunc
    sdf_trunc = args.sdf_trunc

    print(f"\nTSDF parameters (MILo DTU style):")
    print(f"  Voxel size: {voxel_size}")
    print(f"  Depth truncation: {depth_trunc}")
    print(f"  SDF truncation: {sdf_trunc}")
    print(f"  Depth ratio: {args.depth_ratio} (1.0 = median only)")

    # 提取 mesh
    mesh = extractor.extract_mesh(
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc
    )

    # 保存原始 mesh
    mesh_name = f"mesh_tsdf_v{voxel_size}.ply"
    mesh_path = model_path / mesh_name
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"\nMesh saved: {mesh_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.triangles)}")

    # 后处理
    if not args.skip_post_process:
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        post_mesh_path = model_path / mesh_name.replace('.ply', '_post.ply')
        o3d.io.write_triangle_mesh(str(post_mesh_path), mesh_post)
        print(f"\nPost-processed mesh saved: {post_mesh_path}")

        # 用于评估的 mesh
        eval_mesh_path = post_mesh_path
    else:
        eval_mesh_path = mesh_path
        mesh_post = mesh

    # [MESH-CULL] 使用相机 mask 过滤 mesh 顶点（参考 MILo）
    mesh_verts = np.asarray(mesh_post.vertices)
    mesh_faces = np.asarray(mesh_post.triangles)
    if args.cull_mesh:
        try:
            from utils.mesh_cull import cull_mesh_with_cameras
            print(f"\nCulling mesh with camera masks (dilation={args.cull_dilation})...")
            mesh_verts, mesh_faces = cull_mesh_with_cameras(
                mesh_verts, mesh_faces, train_cameras,
                dilation_radius=args.cull_dilation
            )
            # 保存 culled mesh
            culled_mesh = o3d.geometry.TriangleMesh()
            culled_mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
            culled_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
            culled_mesh_path = model_path / mesh_name.replace('.ply', '_culled.ply')
            o3d.io.write_triangle_mesh(str(culled_mesh_path), culled_mesh)
            print(f"Culled mesh saved: {culled_mesh_path}")
        except Exception as e:
            print(f"Warning: mesh culling failed: {e}")
            print("Continuing without culling...")

    # [DTU-COORD] 坐标变换：训练坐标系 -> 原始世界坐标系
    # 需要应用 inv(pca_transform) 和 scale_mat

    inv_transform = np.linalg.inv(scene_info.transform)
    verts_hom = np.hstack([mesh_verts, np.ones((mesh_verts.shape[0], 1))])
    verts_colmap = (inv_transform @ verts_hom.T).T[:, :3]

    # 检查是否有 scale_mat（DTU 数据集）
    cameras_npz_path = Path(config.dataset_path) / 'cameras.npz'
    if cameras_npz_path.exists():
        print(f"\nApplying scale_mat transform for DTU...")
        camera_dict = np.load(str(cameras_npz_path))
        scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
        verts_hom = np.hstack([verts_colmap, np.ones((verts_colmap.shape[0], 1))])
        verts_world = (scale_mat @ verts_hom.T).T[:, :3]
    else:
        verts_world = verts_colmap

    # 保存世界坐标系的 mesh
    mesh_world = o3d.geometry.TriangleMesh()
    mesh_world.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh_world.triangles = o3d.utility.Vector3iVector(mesh_faces)

    world_mesh_path = model_path / mesh_name.replace('.ply', '_world.ply')
    o3d.io.write_triangle_mesh(str(world_mesh_path), mesh_world)
    print(f"World-coordinate mesh saved: {world_mesh_path}")

    # DTU 评估
    if args.eval_dtu:
        import subprocess

        src_path = os.path.normpath(str(config.dataset_path))
        scan_folder = os.path.basename(src_path)
        dataset_root = os.path.dirname(src_path)

        # 提取 scan_id
        scan_digits = ''.join([c for c in scan_folder if c.isdigit()])
        scan_id = scan_digits if scan_digits else '37'

        # 检查是否为 DTU 数据集
        obs_mask_path = os.path.join(dataset_root, 'ObsMask', f'ObsMask{scan_id}_10.mat')
        is_dtu = os.path.exists(obs_mask_path) or cameras_npz_path.exists()

        if not is_dtu:
            print(f"\nSkipping DTU eval: not a DTU dataset")
        else:
            script_path = Path(__file__).parent.parent / 'eval_dtu_vanilla' / 'evaluate_single_scene.py'
            eval_outdir = model_path / 'eval_dtu_tsdf'
            eval_outdir.mkdir(exist_ok=True, parents=True)

            cmd = [
                sys.executable, str(script_path),
                '--input_mesh', str(world_mesh_path),
                '--scan_id', str(scan_id),
                '--output_dir', str(eval_outdir),
                '--mask_dir', dataset_root,
                '--DTU', dataset_root,
            ]
            print(f'\n[EVAL-DTU] Running: {" ".join(cmd)}')
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
