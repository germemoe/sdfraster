#!/usr/bin/env python3
"""
诊断梯度计算问题
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/data/jk/project/radmesh')

def test_gradient():
    from models.ingp_color import Model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟用户的实际配置
    n_points = 1000
    vertices = torch.randn(n_points, 3) * 0.5
    ext_vertices = torch.randn(100, 3) * 2
    center = vertices.mean(dim=0).to(device)
    scaling = torch.tensor(2.38)

    # 使用和用户相同的 sdf_bias
    sdf_bias = 0.4687

    print(f"=== 配置 ===")
    print(f"scene_scaling: {scaling}")
    print(f"sdf_bias: {sdf_bias:.4f}")
    print(f"sphere_radius_world: {sdf_bias * scaling:.4f}")

    model = Model(
        vertices.to(device),
        ext_vertices,
        center,
        scaling,
        sdf_bias=sdf_bias,
        use_contraction=False
    )

    # 直接复制 verify_geometric_init 的采样逻辑
    print(f"\n=== 复现 verify_geometric_init 采样 ===")
    sample_radius = sdf_bias * 2
    grid_size = int(10000 ** (1/3))
    lin = torch.linspace(-sample_radius, sample_radius, grid_size, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing='ij')
    sample_points_normalized = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    sample_dist = sample_points_normalized.norm(dim=-1)
    sample_mask = sample_dist < sample_radius
    sample_points_normalized = sample_points_normalized[sample_mask]
    print(f"采样点数: {sample_points_normalized.shape[0]}")
    print(f"采样半径 (归一化): {sample_radius:.4f}")

    sample_points_world = sample_points_normalized * float(scaling) + center

    # 分析采样点的分布
    print(f"\n采样点分布 (归一化坐标):")
    print(f"  min: {sample_points_normalized.min(dim=0).values.tolist()}")
    print(f"  max: {sample_points_normalized.max(dim=0).values.tolist()}")

    # 计算梯度
    _, gradients, _ = model.compute_sdf_gradients(sample_points_world, mode="numerical", taps=6)
    grad_norms = gradients.norm(dim=-1)

    print(f"\n=== 梯度统计 ===")
    print(f"梯度范数: mean={grad_norms.mean():.4f}, std={grad_norms.std():.4f}")
    print(f"Eikonal loss: {((grad_norms - 1)**2).mean():.6f}")

    # 检查梯度范数的分布
    print(f"\n梯度范数分布:")
    print(f"  < 0.5: {(grad_norms < 0.5).sum().item()} ({100*(grad_norms < 0.5).float().mean():.1f}%)")
    print(f"  0.5-0.8: {((grad_norms >= 0.5) & (grad_norms < 0.8)).sum().item()}")
    print(f"  0.8-1.2: {((grad_norms >= 0.8) & (grad_norms < 1.2)).sum().item()}")
    print(f"  > 1.2: {(grad_norms > 1.2).sum().item()} ({100*(grad_norms > 1.2).float().mean():.1f}%)")

    # 检查低梯度范数的点
    low_grad_mask = grad_norms < 0.5
    if low_grad_mask.any():
        low_grad_points = sample_points_normalized[low_grad_mask]
        print(f"\n低梯度范数点的位置:")
        print(f"  归一化坐标范数: mean={low_grad_points.norm(dim=-1).mean():.4f}")

    # 对比：只在小范围内采样
    print(f"\n=== 对比：小范围采样 (半径=sdf_bias*0.5) ===")
    small_radius = sdf_bias * 0.5
    small_mask = sample_points_normalized.norm(dim=-1) < small_radius
    small_points_world = sample_points_world[small_mask]
    _, small_grad, _ = model.compute_sdf_gradients(small_points_world, mode="numerical", taps=6)
    small_grad_norms = small_grad.norm(dim=-1)
    print(f"采样点数: {small_points_world.shape[0]}")
    print(f"梯度范数: mean={small_grad_norms.mean():.4f}, std={small_grad_norms.std():.4f}")

    # 对比：使用 autograd
    print(f"\n=== 对比：使用 Autograd ===")
    sample_ag = sample_points_world[:1000].clone().requires_grad_(True)
    sdf_ag = model.query_vert_sdf(sample_ag)
    grad_ag = torch.autograd.grad(sdf_ag.sum(), sample_ag)[0]
    grad_ag_norms = grad_ag.norm(dim=-1)
    print(f"Autograd 梯度范数: mean={grad_ag_norms.mean():.4f}, std={grad_ag_norms.std():.4f}")


if __name__ == "__main__":
    test_gradient()
