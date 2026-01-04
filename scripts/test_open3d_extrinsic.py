#!/usr/bin/env python3
"""
验证 Open3D TSDF extrinsic 约定的脚本

测试 W2C vs C2W 哪个是正确的。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import math
from argparse import ArgumentParser
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from data import loader
from utils.args import Args
from utils.train_util import render


def test_extrinsic_convention():
    """测试 Open3D 期望的 extrinsic 是 W2C 还是 C2W"""

    print("Testing Open3D extrinsic convention...")
    print("="*60)

    # 创建一个简单的测试场景
    # 相机在 (0, 0, 2) 朝向 -Z，场景中心在原点

    # 创建一个简单的深度图：中心有一个深度为 1.0 的平面
    H, W = 480, 640
    depth = np.ones((H, W), dtype=np.float32) * 1.0  # 深度 1.0

    # 创建 RGB 图像
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 128

    # 相机内参
    fx, fy = 500, 500
    cx, cy = W/2, H/2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # 测试两种 extrinsic 约定
    # 相机位置: (0, 0, 2)，朝向 -Z

    # W2C: 将世界坐标变换到相机坐标
    # 相机坐标系中，相机在原点，朝向 +Z
    # 所以 W2C = [R | t]，其中 R=I（假设没有旋转），t = (0, 0, -2)
    W2C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -2],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # C2W: W2C 的逆
    C2W = np.linalg.inv(W2C)

    print("Camera at (0, 0, 2), looking at -Z")
    print(f"\nW2C (world to camera):\n{W2C}")
    print(f"\nC2W (camera to world):\n{C2W}")

    # 创建 RGBD 图像
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False
    )

    # 测试 W2C
    print("\n" + "="*60)
    print("Test 1: Using W2C as extrinsic")
    volume1 = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    volume1.integrate(rgbd, intrinsic, W2C)
    mesh1 = volume1.extract_triangle_mesh()
    if len(mesh1.vertices) > 0:
        verts1 = np.asarray(mesh1.vertices)
        print(f"  Mesh vertices: {len(verts1)}")
        print(f"  Z range: [{verts1[:, 2].min():.3f}, {verts1[:, 2].max():.3f}]")
        print(f"  Expected: Z ≈ 1.0 (depth from camera at Z=2)")
    else:
        print("  No mesh extracted!")

    # 测试 C2W
    print("\n" + "="*60)
    print("Test 2: Using C2W as extrinsic")
    volume2 = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    volume2.integrate(rgbd, intrinsic, C2W)
    mesh2 = volume2.extract_triangle_mesh()
    if len(mesh2.vertices) > 0:
        verts2 = np.asarray(mesh2.vertices)
        print(f"  Mesh vertices: {len(verts2)}")
        print(f"  Z range: [{verts2[:, 2].min():.3f}, {verts2[:, 2].max():.3f}]")
        print(f"  Expected: Z ≈ 1.0 (depth from camera at Z=2)")
    else:
        print("  No mesh extracted!")

    print("\n" + "="*60)
    print("Conclusion:")
    print("  If W2C gives Z ≈ 1.0, then Open3D expects W2C")
    print("  If C2W gives Z ≈ 1.0, then Open3D expects C2W")


def main():
    test_extrinsic_convention()


if __name__ == "__main__":
    main()
