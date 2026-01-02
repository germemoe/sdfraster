#!/usr/bin/env python3
"""
分析渲染法线和中位深度转法线的差异
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, '/data/jk/project/radmesh')

from scene import Scene
from scene.sdf_gaussian_model import SDFGaussianModel
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser, Namespace
from gaussian_splatting_renderer import render_tetra
import matplotlib.pyplot as plt

def depth_to_normal(depth, fx, fy, cx, cy):
    """从深度图计算法线（view space）"""
    H, W = depth.shape

    # 创建像素坐标网格
    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')

    # 深度图的梯度
    dz_dx = torch.zeros_like(depth)
    dz_dy = torch.zeros_like(depth)

    # 中心差分，步长2
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2

    # 转换到3D空间的梯度
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    # dx/du = z / fx
    # dy/dv = z / fy

    z = depth
    dx_du = z / fx
    dy_dv = z / fy

    # 法线 = (-dz/dx * dx/du, -dz/dy * dy/dv, 1) 然后归一化
    # 但这里我们用更简单的方式：直接用像素空间的梯度
    nx = -dz_dx * fx / z
    ny = -dz_dy * fy / z
    nz = torch.ones_like(depth)

    # 归一化
    norm = torch.sqrt(nx**2 + ny**2 + nz**2 + 1e-8)
    nx = nx / norm
    ny = ny / norm
    nz = -nz / norm  # 负z朝向相机

    normal = torch.stack([nx, ny, nz], dim=-1)
    return normal

def main():
    # 加载模型
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    args = parser.parse_args([
        '-s', '/storage/jk/dataset/DTU/scan37',
        '-m', '/home/jk/data/project/radmesh/output/sdf/DTU/test/just_tv1e-4_normwdmed_meshdmed_hash22',
        '--eval'
    ])

    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    gaussians = SDFGaussianModel(dataset.sh_degree, pipe.compute_mlp_normal)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)

    # 选一个相机
    camera = scene.getTrainCameras()[8]

    # 渲染
    with torch.no_grad():
        render_pkg = render_tetra(camera, gaussians, pipe, torch.zeros(3).cuda(),
                                  render_normal=True, render_depth=True)

    rendered_normal = render_pkg["normal"]  # [3, H, W]
    depth_med = render_pkg["depth_med"]     # [1, H, W]
    alpha = render_pkg["alpha"]             # [1, H, W]

    # 变换渲染法线到view space
    R = camera.world_view_transform[:3, :3].cuda()  # [3, 3]

    # rendered_normal 是 world space，转到 view space
    H, W = rendered_normal.shape[1], rendered_normal.shape[2]
    n_world = rendered_normal.permute(1, 2, 0).reshape(-1, 3)  # [N, 3]
    n_view = n_world @ R  # [N, 3]
    rendered_normal_view = n_view.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]

    # 从中位深度计算法线
    fx = camera.fx if hasattr(camera, 'fx') else W / (2 * np.tan(camera.FoVx / 2))
    fy = camera.fy if hasattr(camera, 'fy') else H / (2 * np.tan(camera.FoVy / 2))
    cx = camera.cx if hasattr(camera, 'cx') else W / 2
    cy = camera.cy if hasattr(camera, 'cy') else H / 2

    depth_med_normal = depth_to_normal(depth_med[0], fx, fy, cx, cy)  # [H, W, 3]
    depth_med_normal = depth_med_normal.permute(2, 0, 1)  # [3, H, W]

    # 创建有效像素mask
    mask = (alpha[0] > 0.5).float()

    # 归一化渲染法线（确保单位长度）
    rendered_norm = torch.norm(rendered_normal_view, dim=0, keepdim=True)
    rendered_normal_unit = rendered_normal_view / (rendered_norm + 1e-8)

    # 提取有效区域
    valid = mask > 0.5

    rn = rendered_normal_unit.permute(1, 2, 0)[valid]  # [N, 3]
    dn = depth_med_normal.permute(1, 2, 0)[valid]       # [N, 3]

    print("=" * 60)
    print("详细分析：渲染法线 vs 中位深度法线")
    print("=" * 60)

    print(f"\n有效像素数: {rn.shape[0]}")

    # 1. 各分量统计
    print("\n--- 各分量均值和标准差 ---")
    print(f"{'分量':<8} {'渲染法线':<20} {'深度法线':<20} {'差值':<15}")
    for i, name in enumerate(['x', 'y', 'z']):
        r_mean, r_std = rn[:, i].mean().item(), rn[:, i].std().item()
        d_mean, d_std = dn[:, i].mean().item(), dn[:, i].std().item()
        diff = r_mean - d_mean
        print(f"{name:<8} {r_mean:>8.4f} ± {r_std:<8.4f} {d_mean:>8.4f} ± {d_std:<8.4f} {diff:>+8.4f}")

    # 2. 角度差异
    dot = (rn * dn).sum(dim=-1).clamp(-1, 1)
    angle = torch.acos(dot) * 180 / np.pi  # 角度（度）

    print(f"\n--- 角度差异统计 ---")
    print(f"角度差异: mean={angle.mean().item():.2f}°, std={angle.std().item():.2f}°")
    print(f"角度差异: median={angle.median().item():.2f}°")
    print(f"角度差异: max={angle.max().item():.2f}°")
    print(f"角度 < 10°: {(angle < 10).float().mean().item()*100:.1f}%")
    print(f"角度 < 20°: {(angle < 20).float().mean().item()*100:.1f}%")
    print(f"角度 < 30°: {(angle < 30).float().mean().item()*100:.1f}%")

    # 3. 法线长度（渲染法线累积前的长度）
    rendered_len = rendered_norm[0][valid]
    print(f"\n--- 渲染法线长度（归一化前）---")
    print(f"长度: mean={rendered_len.mean().item():.4f}, std={rendered_len.std().item():.4f}")
    print(f"长度: min={rendered_len.min().item():.4f}, max={rendered_len.max().item():.4f}")

    # 4. 分析差异的来源
    # 计算如果只是z分量不同，dot product会是多少
    # dot = rx*dx + ry*dy + rz*dz
    dot_xy = rn[:, 0] * dn[:, 0] + rn[:, 1] * dn[:, 1]
    dot_z = rn[:, 2] * dn[:, 2]

    print(f"\n--- Dot product 分解 ---")
    print(f"xy贡献: mean={dot_xy.mean().item():.4f}")
    print(f"z贡献:  mean={dot_z.mean().item():.4f}")
    print(f"总dot:  mean={dot.mean().item():.4f}")

    # 5. 看看差异大的区域在哪
    print(f"\n--- 差异大的区域分析 ---")

    # 按角度差异分组
    small_diff = angle < 15
    medium_diff = (angle >= 15) & (angle < 30)
    large_diff = angle >= 30

    print(f"小差异 (<15°): {small_diff.float().mean().item()*100:.1f}%")
    print(f"中差异 (15-30°): {medium_diff.float().mean().item()*100:.1f}%")
    print(f"大差异 (>30°): {large_diff.float().mean().item()*100:.1f}%")

    if large_diff.sum() > 0:
        print(f"\n大差异区域的法线特征:")
        rn_large = rn[large_diff]
        dn_large = dn[large_diff]
        print(f"  渲染法线 z: mean={rn_large[:, 2].mean().item():.4f}")
        print(f"  深度法线 z: mean={dn_large[:, 2].mean().item():.4f}")

    # 6. 可视化
    print(f"\n--- 保存可视化 ---")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # 渲染法线各分量
    for i, (name, ax) in enumerate(zip(['x', 'y', 'z'], axes[0, :3])):
        data = rendered_normal_unit[i].cpu().numpy()
        im = ax.imshow(data, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'Rendered Normal {name}')
        plt.colorbar(im, ax=ax)

    # 深度法线各分量
    for i, (name, ax) in enumerate(zip(['x', 'y', 'z'], axes[1, :3])):
        data = depth_med_normal[i].cpu().numpy()
        im = ax.imshow(data, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'DepthMed Normal {name}')
        plt.colorbar(im, ax=ax)

    # 差异
    for i, (name, ax) in enumerate(zip(['x', 'y', 'z'], axes[2, :3])):
        diff = (rendered_normal_unit[i] - depth_med_normal[i]).cpu().numpy()
        im = ax.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax.set_title(f'Diff {name} (Rendered - Depth)')
        plt.colorbar(im, ax=ax)

    # 角度差异图
    angle_map = torch.zeros(H, W, device=angle.device)
    angle_map[valid] = angle
    ax = axes[0, 3]
    im = ax.imshow(angle_map.cpu().numpy(), cmap='hot', vmin=0, vmax=45)
    ax.set_title('Angle Difference (degrees)')
    plt.colorbar(im, ax=ax)

    # 渲染法线长度
    len_map = torch.zeros(H, W, device=rendered_len.device)
    len_map[valid] = rendered_len
    ax = axes[1, 3]
    im = ax.imshow(len_map.cpu().numpy(), cmap='viridis', vmin=0.8, vmax=1.2)
    ax.set_title('Rendered Normal Length')
    plt.colorbar(im, ax=ax)

    # Dot product
    dot_map = torch.zeros(H, W, device=dot.device)
    dot_map[valid] = dot
    ax = axes[2, 3]
    im = ax.imshow(dot_map.cpu().numpy(), cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_title('Dot Product')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = '/home/jk/data/project/radmesh/output/sdf/DTU/test/just_tv1e-4_normwdmed_meshdmed_hash22/debug_normal_diff.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")

    # 7. 分析是否是"倍数"问题
    print(f"\n--- 检查是否是尺度/倍数问题 ---")

    # 如果渲染法线和深度法线只是尺度不同，那么它们应该是平行的
    # 检查 rn / dn 的比值是否一致

    # 避免除零，只看z分量（通常最大）
    ratio_z = rn[:, 2] / (dn[:, 2] + 1e-8)
    valid_ratio = (dn[:, 2].abs() > 0.1)  # 只看z分量较大的区域

    if valid_ratio.sum() > 0:
        ratio_z_valid = ratio_z[valid_ratio]
        print(f"z分量比值 (rendered/depth): mean={ratio_z_valid.mean().item():.4f}, std={ratio_z_valid.std().item():.4f}")

    # 8. 检查法线方向是否一致但幅度不同
    # 计算 rn 在 dn 方向上的投影长度
    proj_len = dot  # 因为 dn 已经是单位向量
    print(f"\n渲染法线在深度法线方向的投影: mean={proj_len.mean().item():.4f}")

    # 渲染法线垂直于深度法线的分量
    perp_len = torch.sqrt(1 - dot**2 + 1e-8)
    print(f"渲染法线垂直分量大小: mean={perp_len.mean().item():.4f}")

if __name__ == '__main__':
    main()
