"""Debug script for analyzing normal_depth_med_loss.

Loads a trained model and analyzes the relationship between:
- Rendered normal (from Gaussian splatting)
- Depth-derived normal (from depth_to_normal)
- Depth_med-derived normal (from median depth)

Usage:
    python scripts/debug_normal_loss.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

from data import loader
from models.ingp_color import Model
from utils.train_util import render
from utils.geometry_utils import depth_to_normal, compute_normal_consistency_loss

# ============ Config ============
CKPT_DIR = Path("/home/jk/data/project/radmesh/output/sdf/DTU/test/just_tv1e-4_normwdmed_meshdmed_hash22")
OUTPUT_DIR = CKPT_DIR / "debug_normal"
SAMPLE_CAM_IDX = 8  # Which camera to use for visualization

device = torch.device('cuda')

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load config
    with open(CKPT_DIR / "config.json") as f:
        config = json.load(f)

    dataset_path = Path(config["dataset_path"])
    print(f"Dataset path: {dataset_path}")

    # Load dataset
    train_cameras, test_cameras, scene_info = loader.load_dataset(
        dataset_path,
        config["image_folder"],
        data_device='cpu',
        eval=config["eval"],
        resolution=config["resolution"]
    )

    # Load model (load_ckpt expects directory, not file)
    print(f"Loading model from {CKPT_DIR}")
    model = Model.load_ckpt(CKPT_DIR, device)
    model.eval()

    # Get sample camera
    camera = train_cameras[SAMPLE_CAM_IDX]
    gt_mask = camera.gt_alpha_mask.cuda() if hasattr(camera, 'gt_alpha_mask') else None

    print(f"\n{'='*60}")
    print(f"Analyzing camera {SAMPLE_CAM_IDX}")
    print(f"Image size: {camera.image_width} x {camera.image_height}")
    print(f"{'='*60}\n")

    # Render
    with torch.no_grad():
        render_pkg = render(camera, model, scene_scaling=model.scene_scaling,
                           tile_size=config["tile_size"], min_t=config["min_t"])

    rendered_normal_world = render_pkg['normal']  # (3, H, W) world space
    depth = render_pkg['depth']  # (H, W)
    depth_med = render_pkg['depth_med']  # (H, W)
    alpha = render_pkg['alpha']  # (H, W)

    # Transform rendered normal to view space
    R = camera.world_view_transform[:3, :3].to(device)
    rendered_normal_view = rendered_normal_world.permute(1, 2, 0) @ R  # (H, W, 3)
    rendered_normal_view = F.normalize(rendered_normal_view, dim=-1)
    rendered_normal_view = rendered_normal_view.permute(2, 0, 1)  # (3, H, W)

    # Compute depth-derived normals
    depth_normal = depth_to_normal(camera, depth)  # (3, H, W)
    depth_med_normal = depth_to_normal(camera, depth_med)  # (3, H, W)

    # Build valid mask
    alpha_mask = alpha > 0.5
    if gt_mask is not None:
        gt_mask_2d = gt_mask.squeeze() > 0.5 if gt_mask.dim() == 3 else gt_mask > 0.5
        valid_mask = alpha_mask & gt_mask_2d
    else:
        valid_mask = alpha_mask

    # ============ Analysis ============
    print("="*60)
    print("COORDINATE SYSTEM DEBUG")
    print("="*60)

    # Check the world_view_transform
    print(f"\nworld_view_transform:\n{camera.world_view_transform}")
    print(f"\nworld_view_transform[:3,:3] (R):\n{R.cpu().numpy()}")

    # Check camera position
    cam_pos = camera.camera_center
    print(f"\nCamera position (world): {cam_pos}")

    # Check rendered normal in world space first
    rn_world = rendered_normal_world.permute(1, 2, 0)[valid_mask]  # (N, 3)
    print(f"\n--- Rendered normal in WORLD space ---")
    print(f"x: mean={rn_world[:, 0].mean():.4f}, std={rn_world[:, 0].std():.4f}")
    print(f"y: mean={rn_world[:, 1].mean():.4f}, std={rn_world[:, 1].std():.4f}")
    print(f"z: mean={rn_world[:, 2].mean():.4f}, std={rn_world[:, 2].std():.4f}")

    # 验证：不做变换直接比较
    print(f"\n--- 直接比较（不变换rendered normal）---")
    dot_no_transform = (rn_world * dn_med).sum(dim=-1) if 'dn_med' in dir() else None

    # 用不同的变换方式测试
    print(f"\n--- 测试不同的坐标变换 ---")

    # 方式1: 当前方式 n_view = n_world @ R
    rn_v1 = rendered_normal_world.permute(1, 2, 0) @ R
    rn_v1 = F.normalize(rn_v1, dim=-1)
    dot_v1 = (rn_v1[valid_mask] * depth_med_normal.permute(1, 2, 0)[valid_mask]).sum(dim=-1).mean()
    print(f"方式1 (n @ R): dot={dot_v1:.4f}")

    # 方式2: n_view = n_world @ R.T
    rn_v2 = rendered_normal_world.permute(1, 2, 0) @ R.T
    rn_v2 = F.normalize(rn_v2, dim=-1)
    dot_v2 = (rn_v2[valid_mask] * depth_med_normal.permute(1, 2, 0)[valid_mask]).sum(dim=-1).mean()
    print(f"方式2 (n @ R.T): dot={dot_v2:.4f}")

    # 方式3: 不变换（假设已经是view space）
    rn_v3 = rendered_normal_world.permute(1, 2, 0)
    dot_v3 = (rn_v3[valid_mask] * depth_med_normal.permute(1, 2, 0)[valid_mask]).sum(dim=-1).mean()
    print(f"方式3 (不变换): dot={dot_v3:.4f}")

    # 方式4: flip z
    rn_v4 = rendered_normal_view.clone()
    rn_v4[2] = -rn_v4[2]
    rn_v4 = F.normalize(rn_v4, dim=0)
    dot_v4 = (rn_v4.permute(1, 2, 0)[valid_mask] * depth_med_normal.permute(1, 2, 0)[valid_mask]).sum(dim=-1).mean()
    print(f"方式4 (flip z): dot={dot_v4:.4f}")

    print(f"\n最佳方式应该是 dot 最接近 1.0 的")

    # 测试不同的有限差分步长
    print(f"\n--- 测试不同的有限差分步长 ---")

    def depth_to_normal_custom(camera, depth_map, step=2):
        """用自定义步长计算法线"""
        from utils.geometry_utils import depths_to_points
        points = depths_to_points(camera, depth_map)
        # step=1 或 step=2
        if step == 1:
            dx = points[:, 1:, :-1] - points[:, :-1, :-1]
            dy = points[:, :-1, 1:] - points[:, :-1, :-1]
            normal = torch.cross(dx, dy, dim=0)
            normal = F.normalize(normal, dim=0)
            output = torch.zeros_like(points)
            output[:, :-1, :-1] = normal
        else:  # step=2
            dx = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
            dy = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
            normal = torch.cross(dx, dy, dim=0)
            normal = F.normalize(normal, dim=0)
            output = torch.zeros_like(points)
            output[:, 1:-1, 1:-1] = normal
        return output

    for step in [1, 2]:
        dn_step = depth_to_normal_custom(camera, depth_med, step=step)
        dn_step_valid = dn_step.permute(1, 2, 0)[valid_mask]
        z_mean = dn_step_valid[:, 2].mean()
        dot_mean = (rn_v1[valid_mask] * dn_step_valid).sum(dim=-1).mean()
        print(f"  step={step}: z_mean={z_mean:.4f}, dot={dot_mean:.4f}")

    # 检查深度图的梯度大小
    print(f"\n--- 深度图梯度分析 ---")
    depth_valid = depth_med[valid_mask]
    print(f"DepthMed: min={depth_valid.min():.4f}, max={depth_valid.max():.4f}, std={depth_valid.std():.4f}")

    # 计算深度的空间梯度
    depth_dx = depth_med[2:, 1:-1] - depth_med[:-2, 1:-1]
    depth_dy = depth_med[1:-1, 2:] - depth_med[1:-1, :-2]
    depth_grad_mag = torch.sqrt(depth_dx**2 + depth_dy**2)
    print(f"Depth gradient magnitude: mean={depth_grad_mag.mean():.6f}, max={depth_grad_mag.max():.6f}")

    # 只看中心区域 (排除边缘)
    print(f"\n--- 中心区域分析（排除边缘影响）---")
    H, W = depth_med.shape
    margin = 100
    center_mask = torch.zeros_like(valid_mask)
    center_mask[margin:H-margin, margin:W-margin] = True
    center_valid = valid_mask & center_mask

    rn_center = rn_v1[center_valid]  # rendered normal (view space)
    dn_center = depth_med_normal.permute(1, 2, 0)[center_valid]
    dot_center = (rn_center * dn_center).sum(dim=-1)
    print(f"Center region ({center_valid.sum()} pixels):")
    print(f"  Rendered z mean: {rn_center[:, 2].mean():.4f}")
    print(f"  DepthMed z mean: {dn_center[:, 2].mean():.4f}")
    print(f"  Dot mean: {dot_center.mean():.4f}")
    print(f"  Loss: {(1 - dot_center).mean():.4f}")

    # 检查 alpha 分布
    print(f"\n--- Alpha 分布分析 ---")
    alpha_valid = alpha[valid_mask]
    print(f"Alpha: min={alpha_valid.min():.4f}, max={alpha_valid.max():.4f}, mean={alpha_valid.mean():.4f}")
    print(f"Alpha > 0.9: {(alpha_valid > 0.9).float().mean():.2%}")
    print(f"Alpha > 0.95: {(alpha_valid > 0.95).float().mean():.2%}")
    print(f"Alpha > 0.99: {(alpha_valid > 0.99).float().mean():.2%}")

    # 只看高 alpha 区域
    high_alpha_mask = valid_mask & (alpha > 0.95)
    if high_alpha_mask.sum() > 1000:
        rn_ha = rn_v1[high_alpha_mask]
        dn_ha = depth_med_normal.permute(1, 2, 0)[high_alpha_mask]
        dot_ha = (rn_ha * dn_ha).sum(dim=-1)
        print(f"\nHigh alpha region (alpha > 0.95, {high_alpha_mask.sum()} pixels):")
        print(f"  Rendered z mean: {rn_ha[:, 2].mean():.4f}")
        print(f"  DepthMed z mean: {dn_ha[:, 2].mean():.4f}")
        print(f"  Dot mean: {dot_ha.mean():.4f}")
        print(f"  Loss: {(1 - dot_ha).mean():.4f}")

    # 验证：如果渲染法线是正确的，用它来计算深度应该与实际深度一致
    # 对于面向相机的表面：z_normal ≈ -1 / sqrt(1 + (dz/dx)^2 + (dz/dy)^2)
    # 从法线反推深度梯度：dz/dx = -nx/nz, dz/dy = -ny/nz
    print(f"\n--- 法线一致性验证 ---")

    # 从渲染法线估计的深度梯度
    rn_full = rn_v1  # (H, W, 3)
    nz = rn_full[:, :, 2].clamp(max=-0.01)  # 确保 z < 0
    estimated_dzdx = -rn_full[:, :, 0] / nz
    estimated_dzdy = -rn_full[:, :, 1] / nz

    # 从实际深度计算的深度梯度 (在像素坐标系下)
    from utils.geometry_utils import depths_to_points
    points = depths_to_points(camera, depth_med)  # (3, H, W)

    # 实际 dz/dx 和 dz/dy (使用相同的 2-step 差分)
    actual_dzdx = (points[2, 1:-1, 2:] - points[2, 1:-1, :-2]) / (points[0, 1:-1, 2:] - points[0, 1:-1, :-2] + 1e-8)
    actual_dzdy = (points[2, 2:, 1:-1] - points[2, :-2, 1:-1]) / (points[1, 2:, 1:-1] - points[1, :-2, 1:-1] + 1e-8)

    # 比较中心区域
    inner_mask = center_valid[1:-1, 1:-1]
    est_dzdx_inner = estimated_dzdx[1:-1, 1:-1][inner_mask]
    est_dzdy_inner = estimated_dzdy[1:-1, 1:-1][inner_mask]
    act_dzdx_inner = actual_dzdx[inner_mask]
    act_dzdy_inner = actual_dzdy[inner_mask]

    print(f"dz/dx: estimated mean={est_dzdx_inner.mean():.4f}, actual mean={act_dzdx_inner.mean():.4f}")
    print(f"dz/dy: estimated mean={est_dzdy_inner.mean():.4f}, actual mean={act_dzdy_inner.mean():.4f}")
    print(f"dz/dx correlation: {torch.corrcoef(torch.stack([est_dzdx_inner.flatten()[:10000], act_dzdx_inner.flatten()[:10000]]))[0,1]:.4f}")
    print(f"dz/dy correlation: {torch.corrcoef(torch.stack([est_dzdy_inner.flatten()[:10000], act_dzdy_inner.flatten()[:10000]]))[0,1]:.4f}")

    print("="*60)
    print("NORMAL DIRECTION ANALYSIS")
    print("="*60)

    # Extract valid pixels
    rn = rendered_normal_view.permute(1, 2, 0)[valid_mask]  # (N, 3)
    dn = depth_normal.permute(1, 2, 0)[valid_mask]  # (N, 3)
    dn_med = depth_med_normal.permute(1, 2, 0)[valid_mask]  # (N, 3)

    print(f"\nNumber of valid pixels: {valid_mask.sum().item()}")

    # Z-component analysis (should be negative if pointing towards camera)
    print(f"\n--- Z-component (view space, negative = towards camera) ---")
    print(f"Rendered normal z: mean={rn[:, 2].mean():.4f}, std={rn[:, 2].std():.4f}")
    print(f"Depth normal z:    mean={dn[:, 2].mean():.4f}, std={dn[:, 2].std():.4f}")
    print(f"DepthMed normal z: mean={dn_med[:, 2].mean():.4f}, std={dn_med[:, 2].std():.4f}")

    # Check if directions are consistent
    rn_z_neg = (rn[:, 2] < 0).float().mean()
    dn_z_neg = (dn[:, 2] < 0).float().mean()
    dn_med_z_neg = (dn_med[:, 2] < 0).float().mean()
    print(f"\nFraction with z < 0:")
    print(f"  Rendered:  {rn_z_neg:.2%}")
    print(f"  Depth:     {dn_z_neg:.2%}")
    print(f"  DepthMed:  {dn_med_z_neg:.2%}")

    # Dot product analysis
    print(f"\n--- Dot product analysis ---")
    dot_rn_dn = (rn * dn).sum(dim=-1)
    dot_rn_dn_med = (rn * dn_med).sum(dim=-1)
    dot_dn_dn_med = (dn * dn_med).sum(dim=-1)

    print(f"Rendered vs Depth:    mean={dot_rn_dn.mean():.4f}, std={dot_rn_dn.std():.4f}")
    print(f"Rendered vs DepthMed: mean={dot_rn_dn_med.mean():.4f}, std={dot_rn_dn_med.std():.4f}")
    print(f"Depth vs DepthMed:    mean={dot_dn_dn_med.mean():.4f}, std={dot_dn_dn_med.std():.4f}")

    # Check for opposite directions
    frac_opposite_rn_dn = (dot_rn_dn < 0).float().mean()
    frac_opposite_rn_dn_med = (dot_rn_dn_med < 0).float().mean()
    print(f"\nFraction with dot < 0 (opposite direction):")
    print(f"  Rendered vs Depth:    {frac_opposite_rn_dn:.2%}")
    print(f"  Rendered vs DepthMed: {frac_opposite_rn_dn_med:.2%}")

    # Loss computation
    print(f"\n--- Loss values ---")
    loss_rn_dn = compute_normal_consistency_loss(rendered_normal_view, depth_normal, valid_mask)
    loss_rn_dn_med = compute_normal_consistency_loss(rendered_normal_view, depth_med_normal, valid_mask)
    loss_dn_dn_med = compute_normal_consistency_loss(depth_normal, depth_med_normal, valid_mask)

    print(f"Loss (Rendered vs Depth):    {loss_rn_dn.item():.4f}")
    print(f"Loss (Rendered vs DepthMed): {loss_rn_dn_med.item():.4f}")
    print(f"Loss (Depth vs DepthMed):    {loss_dn_dn_med.item():.4f}")

    # Loss with abs (for comparison)
    loss_rn_dn_med_abs = 1.0 - (rn * dn_med).sum(dim=-1).abs().mean()
    print(f"\nLoss with abs (Rendered vs DepthMed): {loss_rn_dn_med_abs.item():.4f}")

    # ============ Visualization ============
    print(f"\n{'='*60}")
    print("Saving visualizations...")
    print(f"{'='*60}\n")

    def normal_to_vis(n, mask=None):
        """Convert normal (3, H, W) to visualization (H, W, 3) uint8"""
        vis = ((1 - n.permute(1, 2, 0).cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)
        if mask is not None:
            vis[~mask.cpu().numpy()] = 0
        return vis

    def save_comparison(normals_dict, filename, mask=None):
        """Save side-by-side comparison of normals"""
        n_cols = len(normals_dict)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for ax, (name, normal) in zip(axes, normals_dict.items()):
            vis = normal_to_vis(normal, mask)
            ax.imshow(vis)
            ax.set_title(name)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()

    # Save individual normals
    save_comparison({
        'Rendered (view)': rendered_normal_view,
        'Depth Normal': depth_normal,
        'DepthMed Normal': depth_med_normal,
    }, 'normals_comparison.png', valid_mask)

    # Save z-component visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, normal) in zip(axes, [
        ('Rendered z', rendered_normal_view[2]),
        ('Depth z', depth_normal[2]),
        ('DepthMed z', depth_med_normal[2])
    ]):
        z_map = normal.cpu().numpy()
        z_map[~valid_mask.cpu().numpy()] = 0
        im = ax.imshow(z_map, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'{name}\nmean={z_map[valid_mask.cpu().numpy()].mean():.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'z_component.png'), dpi=150)
    plt.close()

    # Save dot product map
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dot_map_dn = (rendered_normal_view * depth_normal).sum(dim=0).cpu().numpy()
    dot_map_dn_med = (rendered_normal_view * depth_med_normal).sum(dim=0).cpu().numpy()

    for ax, (name, dot_map) in zip(axes, [
        ('Rendered vs Depth', dot_map_dn),
        ('Rendered vs DepthMed', dot_map_dn_med)
    ]):
        dot_map[~valid_mask.cpu().numpy()] = 0
        im = ax.imshow(dot_map, cmap='RdYlGn', vmin=-1, vmax=1)
        mean_dot = dot_map[valid_mask.cpu().numpy()].mean()
        ax.set_title(f'{name}\nmean dot={mean_dot:.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dot_product.png'), dpi=150)
    plt.close()

    # Save loss map (per-pixel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    loss_map_dn = (1 - dot_map_dn)
    loss_map_dn_med = (1 - dot_map_dn_med)

    for ax, (name, loss_map) in zip(axes, [
        ('Loss: Rendered vs Depth', loss_map_dn),
        ('Loss: Rendered vs DepthMed', loss_map_dn_med)
    ]):
        loss_map[~valid_mask.cpu().numpy()] = 0
        im = ax.imshow(loss_map, cmap='hot', vmin=0, vmax=2)
        mean_loss = loss_map[valid_mask.cpu().numpy()].mean()
        ax.set_title(f'{name}\nmean loss={mean_loss:.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_map.png'), dpi=150)
    plt.close()

    # Save depth comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    depth_np = depth.cpu().numpy()
    depth_med_np = depth_med.cpu().numpy()

    vmin = min(depth_np[valid_mask.cpu().numpy()].min(), depth_med_np[valid_mask.cpu().numpy()].min())
    vmax = max(depth_np[valid_mask.cpu().numpy()].max(), depth_med_np[valid_mask.cpu().numpy()].max())

    for ax, (name, d) in zip(axes, [('Expected Depth', depth_np), ('Median Depth', depth_med_np)]):
        d_vis = d.copy()
        d_vis[~valid_mask.cpu().numpy()] = 0
        im = ax.imshow(d_vis, cmap='turbo', vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'depth_comparison.png'), dpi=150)
    plt.close()

    print(f"Saved visualizations to {OUTPUT_DIR}")

    # ============ 详细差异分析 ============
    print(f"\n{'='*60}")
    print("DETAILED DIFFERENCE ANALYSIS")
    print(f"{'='*60}")

    # 使用 rn_v1 (view space) 和 dn_med
    rn_view = rn_v1[valid_mask]  # (N, 3) - 渲染法线 view space
    dn_view = depth_med_normal.permute(1, 2, 0)[valid_mask]  # (N, 3) - 深度法线

    # 1. 角度差异
    dot_prod = (rn_view * dn_view).sum(dim=-1).clamp(-1, 1)
    angle_deg = torch.acos(dot_prod) * 180 / np.pi

    print(f"\n--- 角度差异统计 ---")
    print(f"角度差异: mean={angle_deg.mean().item():.2f}°, std={angle_deg.std().item():.2f}°")
    print(f"角度差异: median={angle_deg.median().item():.2f}°")
    print(f"角度差异: max={angle_deg.max().item():.2f}°")
    print(f"角度 < 10°: {(angle_deg < 10).float().mean().item()*100:.1f}%")
    print(f"角度 < 20°: {(angle_deg < 20).float().mean().item()*100:.1f}%")
    print(f"角度 < 30°: {(angle_deg < 30).float().mean().item()*100:.1f}%")

    # 2. 各分量差异
    print(f"\n--- 各分量差异 ---")
    print(f"{'分量':<8} {'渲染法线':<25} {'深度法线':<25} {'差值':<15}")
    for i, name in enumerate(['x', 'y', 'z']):
        r_mean, r_std = rn_view[:, i].mean().item(), rn_view[:, i].std().item()
        d_mean, d_std = dn_view[:, i].mean().item(), dn_view[:, i].std().item()
        diff = r_mean - d_mean
        print(f"{name:<8} {r_mean:>+8.4f} ± {r_std:<12.4f} {d_mean:>+8.4f} ± {d_std:<12.4f} {diff:>+8.4f}")

    # 3. Dot product 分解
    dot_xy = rn_view[:, 0] * dn_view[:, 0] + rn_view[:, 1] * dn_view[:, 1]
    dot_z = rn_view[:, 2] * dn_view[:, 2]
    print(f"\n--- Dot product 分解 ---")
    print(f"xy贡献: mean={dot_xy.mean().item():.4f}")
    print(f"z贡献:  mean={dot_z.mean().item():.4f}")
    print(f"总dot:  mean={dot_prod.mean().item():.4f}")

    # 4. 检查是否是尺度问题
    # 如果渲染法线在归一化前有不同的长度，可能说明 SDF 梯度的问题
    rendered_len = torch.norm(render_pkg['normal'], dim=0)[valid_mask]
    print(f"\n--- 渲染法线长度（归一化前）---")
    print(f"长度: mean={rendered_len.mean().item():.4f}, std={rendered_len.std().item():.4f}")
    print(f"长度: min={rendered_len.min().item():.4f}, max={rendered_len.max().item():.4f}")

    # 5. z分量比值分析
    # 渲染法线 z 均值 -0.85，深度法线 z 均值 -0.61
    # 可能是渲染法线"过于垂直"
    print(f"\n--- z 分量比值分析 ---")
    valid_z = dn_view[:, 2].abs() > 0.1
    if valid_z.sum() > 0:
        ratio_z = rn_view[valid_z, 2] / dn_view[valid_z, 2]
        print(f"z比值 (rendered/depth): mean={ratio_z.mean().item():.4f}, std={ratio_z.std().item():.4f}")
        print(f"z比值: median={ratio_z.median().item():.4f}")

    # 6. 分区域分析差异
    print(f"\n--- 按区域分析 ---")
    small_diff = angle_deg < 15
    medium_diff = (angle_deg >= 15) & (angle_deg < 30)
    large_diff = angle_deg >= 30

    print(f"小差异 (<15°): {small_diff.float().mean().item()*100:.1f}%")
    print(f"中差异 (15-30°): {medium_diff.float().mean().item()*100:.1f}%")
    print(f"大差异 (>30°): {large_diff.float().mean().item()*100:.1f}%")

    # 7. 可视化角度差异分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 角度差异热力图
    angle_map = torch.zeros_like(valid_mask, dtype=torch.float32)
    angle_map[valid_mask] = angle_deg
    ax = axes[0, 0]
    im = ax.imshow(angle_map.cpu().numpy(), cmap='hot', vmin=0, vmax=45)
    ax.set_title('Angle Difference (degrees)')
    plt.colorbar(im, ax=ax)

    # 各分量差异
    for idx, name in enumerate(['x', 'y', 'z']):
        diff_map = torch.zeros_like(valid_mask, dtype=torch.float32)
        diff_values = rn_view[:, idx] - dn_view[:, idx]
        diff_map[valid_mask] = diff_values
        ax = axes[0, 1] if idx == 0 else (axes[0, 2] if idx == 1 else axes[1, 0])
        im = ax.imshow(diff_map.cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax.set_title(f'Diff {name} (Rendered - Depth)')
        plt.colorbar(im, ax=ax)

    # 渲染法线长度
    len_map = torch.zeros_like(valid_mask, dtype=torch.float32)
    len_map[valid_mask] = rendered_len
    ax = axes[1, 1]
    im = ax.imshow(len_map.cpu().numpy(), cmap='viridis', vmin=0.5, vmax=1.5)
    ax.set_title('Rendered Normal Length (before normalize)')
    plt.colorbar(im, ax=ax)

    # z比值
    z_ratio_map = torch.zeros_like(valid_mask, dtype=torch.float32)
    z_ratio_full = torch.zeros(rn_view.shape[0], device=rn_view.device)
    z_ratio_full[valid_z] = ratio_z if valid_z.sum() > 0 else torch.zeros_like(z_ratio_full[valid_z])
    z_ratio_map[valid_mask] = z_ratio_full
    ax = axes[1, 2]
    im = ax.imshow(z_ratio_map.cpu().numpy(), cmap='RdBu', vmin=0.5, vmax=2.0)
    ax.set_title('z ratio (Rendered/Depth)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'detailed_diff.png'), dpi=150)
    plt.close()
    print(f"\nSaved detailed_diff.png")

    # 8. 角度差异直方图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(angle_deg.cpu().numpy(), bins=50, range=(0, 90), edgecolor='black')
    ax.axvline(angle_deg.mean().item(), color='r', linestyle='--', label=f'Mean: {angle_deg.mean().item():.1f}°')
    ax.axvline(angle_deg.median().item(), color='g', linestyle='--', label=f'Median: {angle_deg.median().item():.1f}°')
    ax.set_xlabel('Angle Difference (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Angle Difference between Rendered and DepthMed Normals')
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'angle_histogram.png'), dpi=150)
    plt.close()
    print(f"Saved angle_histogram.png")

    # ============ Recommendation ============
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    if frac_opposite_rn_dn_med > 0.3:
        print("\n[WARNING] >30% of pixels have opposite normal directions!")
        print("This is likely causing the high loss and bumpy surfaces.")
        print("\nPossible fixes:")
        print("1. Flip one of the normals: depth_med_normal = -depth_med_normal")
        print("2. Use abs() in the loss: loss = 1 - |dot(n1, n2)|")
    elif loss_rn_dn_med > 0.15:
        print("\n[INFO] Loss is high but normals are mostly aligned.")
        print("This might be due to noisy median depth.")
        print("\nPossible fixes:")
        print("1. Use expected depth instead of median depth")
        print("2. Lower the loss weight")
    else:
        print("\n[OK] Normal loss looks reasonable.")

if __name__ == "__main__":
    main()
