#!/usr/bin/env python3
"""
用 mesh 渲染深度做 TSDF 融合，排查问题来源。

如果这个脚本生成的 mesh 没有多视角不一致问题，说明问题在体渲染的深度。
如果还有问题，说明问题在相机参数或 TSDF 融合本身。
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import loader
from utils.args import Args


def render_depth_from_mesh(mesh, cameras, image_height, image_width):
    """用 Open3D 从 mesh 渲染深度图"""
    depth_maps = []

    # 创建 Open3D 可视化器（离屏渲染）
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_width, height=image_height, visible=False)
    vis.add_geometry(mesh)

    for cam in tqdm(cameras, desc="Rendering mesh depth"):
        # 设置相机参数
        ctr = vis.get_view_control()

        # 从相机参数计算内参
        fx = image_width / (2 * math.tan(cam.fovx / 2.))
        fy = image_height / (2 * math.tan(cam.fovy / 2.))
        cx = image_width / 2.
        cy = image_height / 2.

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=image_width,
            height=image_height,
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        # 外参：world_view_transform.T
        extrinsic = cam.world_view_transform.T.cpu().numpy()

        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = intrinsic
        param.extrinsic = extrinsic

        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

        # 捕获深度
        depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
        depth_maps.append(depth)

    vis.destroy_window()
    return depth_maps


def render_depth_from_mesh_raycasting(mesh, cameras):
    """用 Open3D RaycastingScene 从 mesh 渲染深度图（更精确）"""
    # 创建 RaycastingScene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    depth_maps = []

    for cam in tqdm(cameras, desc="Raycasting mesh depth"):
        H = cam.image_height
        W = cam.image_width

        # 计算内参
        fx = W / (2 * math.tan(cam.fovx / 2.))
        fy = H / (2 * math.tan(cam.fovy / 2.))
        cx = W / 2.
        cy = H / 2.

        # 生成射线
        # 像素坐标
        y_coords, x_coords = np.meshgrid(
            np.arange(H, dtype=np.float32) + 0.5,
            np.arange(W, dtype=np.float32) + 0.5,
            indexing='ij'
        )

        # 相机空间方向（未归一化）
        dirs_cam = np.stack([
            (x_coords - cx) / fx,
            (y_coords - cy) / fy,
            np.ones_like(x_coords)
        ], axis=-1)  # (H, W, 3)

        # 转换到世界空间
        w2v = cam.world_view_transform.T.cpu().numpy()  # (4, 4)
        v2w = np.linalg.inv(w2v)
        R_v2w = v2w[:3, :3]
        cam_pos = v2w[:3, 3]

        # 世界空间方向
        dirs_world = dirs_cam @ R_v2w.T  # (H, W, 3)
        dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)

        # 射线原点
        origins = np.broadcast_to(cam_pos, dirs_world.shape)

        # 创建射线张量
        rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
        rays = rays.reshape(-1, 6)
        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # 光线追踪
        result = scene.cast_rays(rays_t)
        t_hit = result['t_hit'].numpy().reshape(H, W)

        # t_hit 是沿射线的距离，需要转换为 z-depth
        # z = t * cos(theta) = t / ||d_unnorm||
        d_unnorm_len = np.sqrt(
            ((x_coords - cx) / fx)**2 +
            ((y_coords - cy) / fy)**2 +
            1.0
        )
        z_depth = t_hit / d_unnorm_len

        # 处理未命中的射线
        z_depth[t_hit == np.inf] = 0

        depth_maps.append(z_depth)

    return depth_maps


def tsdf_fusion(depth_maps, cameras, voxel_size=0.002, sdf_trunc=0.01, depth_trunc=8.0):
    """TSDF 融合"""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i, (depth, cam) in tqdm(enumerate(zip(depth_maps, cameras)),
                                 desc="TSDF integration", total=len(depth_maps)):
        H = cam.image_height
        W = cam.image_width

        # 内参
        fx = W / (2 * math.tan(cam.fovx / 2.))
        fy = H / (2 * math.tan(cam.fovy / 2.))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H,
            fx=fx, fy=fy,
            cx=float(W)/2, cy=float(H)/2
        )

        # [VERIFIED] Open3D TSDF integrate 期望 extrinsic 是 W2C（world-to-camera）
        extrinsic = cam.world_view_transform.T.cpu().numpy()  # W2C

        # 创建 RGBD 图像（假颜色）
        color_np = np.ones((H, W, 3), dtype=np.uint8) * 128
        depth_np = depth.astype(np.float32)

        # 应用 mask
        if hasattr(cam, 'gt_alpha_mask') and cam.gt_alpha_mask is not None:
            mask = cam.gt_alpha_mask.cpu().squeeze().numpy()
            if mask.ndim == 2:
                depth_np[mask < 0.5] = 0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_np),
            o3d.geometry.Image(depth_np),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0
        )

        volume.integrate(rgbd, intrinsic, extrinsic)

    return volume.extract_triangle_mesh()


def main():
    parser = ArgumentParser(description="Debug TSDF with mesh-rendered depth")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mesh_file", type=str, default=None,
                        help="Input mesh file. If not specified, will look for mesh_tsdf*.ply")
    parser.add_argument("--voxel_size", type=float, default=0.002)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)

    # 加载配置和数据集
    config = Args.load_from_json(str(model_path / "config.json"))
    train_cameras, _, scene_info = loader.load_dataset(
        config.dataset_path, config.image_folder,
        data_device='cpu', eval=True, resolution=config.resolution
    )

    # 加载 mesh
    if args.mesh_file:
        mesh_file = args.mesh_file
    else:
        # 查找现有的 mesh 文件
        mesh_files = list(model_path.glob("mesh_tsdf*.ply"))
        mesh_files = [f for f in mesh_files if 'world' not in f.name and 'post' not in f.name and 'culled' not in f.name]
        if not mesh_files:
            print("Error: No mesh file found. Please specify --mesh_file")
            return
        mesh_file = str(mesh_files[0])

    print(f"Loading mesh from {mesh_file}...")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.triangles)}")

    # 用 mesh 渲染深度
    print("\nRendering depth from mesh using raycasting...")
    depth_maps = render_depth_from_mesh_raycasting(mesh, train_cameras)

    # 检查深度范围
    print("\nDepth statistics:")
    for i in [0, 10, 20, 30, 40]:
        if i < len(depth_maps):
            d = depth_maps[i]
            valid = d > 0
            if valid.sum() > 0:
                print(f"  Cam{i}: min={d[valid].min():.3f}, max={d[valid].max():.3f}, "
                      f"mean={d[valid].mean():.3f}, valid={valid.sum()}")

    # TSDF 融合
    print("\nRunning TSDF fusion...")
    mesh_fused = tsdf_fusion(depth_maps, train_cameras, voxel_size=args.voxel_size)

    # 保存结果
    output_file = args.output or str(model_path / "mesh_tsdf_from_mesh_depth.ply")
    o3d.io.write_triangle_mesh(output_file, mesh_fused)
    print(f"\nSaved: {output_file}")
    print(f"  Vertices: {len(mesh_fused.vertices)}")
    print(f"  Faces: {len(mesh_fused.triangles)}")

    # 转换到世界坐标系
    mesh_verts = np.asarray(mesh_fused.vertices)
    mesh_faces = np.asarray(mesh_fused.triangles)

    inv_transform = np.linalg.inv(scene_info.transform)
    verts_hom = np.hstack([mesh_verts, np.ones((mesh_verts.shape[0], 1))])
    verts_colmap = (inv_transform @ verts_hom.T).T[:, :3]

    cameras_npz_path = Path(config.dataset_path) / 'cameras.npz'
    if cameras_npz_path.exists():
        camera_dict = np.load(str(cameras_npz_path))
        scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
        verts_hom = np.hstack([verts_colmap, np.ones((verts_colmap.shape[0], 1))])
        verts_world = (scale_mat @ verts_hom.T).T[:, :3]
    else:
        verts_world = verts_colmap

    mesh_world = o3d.geometry.TriangleMesh()
    mesh_world.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh_world.triangles = o3d.utility.Vector3iVector(mesh_faces)

    world_output = output_file.replace('.ply', '_world.ply')
    o3d.io.write_triangle_mesh(world_output, mesh_world)
    print(f"World mesh saved: {world_output}")


if __name__ == "__main__":
    main()
