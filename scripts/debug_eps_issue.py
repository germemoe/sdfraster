#!/usr/bin/env python3
"""
诊断 Eikonal Loss 中 eps 参数的影响

问题: eps=None (auto) 时 loss=0.005，eps=1e-6 时 loss=0.9
原因分析: eps=1e-6 远小于 hash grid 的最小 cell 大小，导致数值梯度接近 0
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/data/jk/project/radmesh')

from models.ingp_color import Model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建测试模型（使用典型配置）
    n_points = 1000
    vertices = torch.randn(n_points, 3) * 0.5
    ext_vertices = torch.randn(100, 3) * 2
    center = vertices.mean(dim=0).to(device)
    scaling = torch.tensor(2.38)  # 典型 scene_scaling
    sdf_bias = 0.4687

    model = Model(
        vertices.to(device),
        ext_vertices,
        center,
        scaling,
        sdf_bias=sdf_bias,
        use_contraction=False,
        base_resolution=64,
        per_level_scale=2,
        L=8
    )

    # 计算关键尺度
    backbone = model.backbone
    finest_resolution = backbone.base_resolution * (backbone.per_level_scale ** (backbone.L - 1))
    auto_eps = float(model.scene_scaling) / finest_resolution
    min_cell_size_01 = 1.0 / finest_resolution  # [0,1] 空间
    min_cell_size_world = float(model.scene_scaling) * 2 / finest_resolution  # 世界空间 (归一化空间是 [-1,1])

    print("\n" + "="*60)
    print("关键尺度参数")
    print("="*60)
    print(f"scene_scaling: {float(model.scene_scaling):.4f}")
    print(f"finest_resolution: {finest_resolution}")
    print(f"min_cell_size ([0,1] 空间): {min_cell_size_01:.6f}")
    print(f"min_cell_size (世界空间): {min_cell_size_world:.6f}")
    print(f"auto_eps (current impl): {auto_eps:.6f}")

    # 测试不同 eps 值
    eps_values = [
        ("auto", auto_eps),
        ("1e-3", 1e-3),
        ("1e-4", 1e-4),
        ("1e-5", 1e-5),
        ("1e-6", 1e-6),
        ("1e-7", 1e-7),
    ]

    # 采样测试点（在球面附近）
    n_samples = 500
    phi = torch.rand(n_samples, device=device) * 2 * np.pi
    cos_theta = torch.rand(n_samples, device=device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)
    sphere_dirs = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta
    ], dim=-1)

    # 在球面上采样（世界坐标）
    sphere_radius_world = sdf_bias * float(scaling)
    test_points = sphere_dirs * sphere_radius_world + center

    print("\n" + "="*60)
    print("不同 eps 值对梯度计算的影响")
    print("="*60)
    print(f"测试点数: {n_samples}")
    print(f"测试点位置: 球面 (radius={sphere_radius_world:.4f})")

    # 首先用 autograd 计算"真实"梯度作为参考
    test_points_ag = test_points.clone().requires_grad_(True)
    sdf_ag = model.query_vert_sdf(test_points_ag)
    grad_ag = torch.autograd.grad(sdf_ag.sum(), test_points_ag)[0]
    grad_ag_norms = grad_ag.norm(dim=-1)
    eikonal_ag = ((grad_ag_norms - 1)**2).mean().item()

    print(f"\n[Autograd 参考]")
    print(f"  梯度范数: mean={grad_ag_norms.mean():.4f}, std={grad_ag_norms.std():.4f}")
    print(f"  Eikonal loss: {eikonal_ag:.6f}")

    print(f"\n[数值梯度 (6-tap)]")
    for name, eps in eps_values:
        _, gradients, _ = model.compute_sdf_gradients(test_points, mode="numerical", taps=6, eps=eps)
        grad_norms = gradients.norm(dim=-1)
        eikonal = ((grad_norms - 1)**2).mean().item()

        # 与 autograd 的差异
        grad_diff = (gradients - grad_ag).norm(dim=-1).mean().item()

        # eps 与 cell size 的比值
        eps_ratio = eps / min_cell_size_world

        print(f"  eps={name:>6} ({eps:.2e}): "
              f"grad_norm={grad_norms.mean():.4f}±{grad_norms.std():.4f}, "
              f"Eikonal={eikonal:.4f}, "
              f"diff_from_AG={grad_diff:.4f}, "
              f"eps/cell={eps_ratio:.3f}")

    print("\n" + "="*60)
    print("结论")
    print("="*60)
    print(f"""
当 eps 太小时（远小于 hash grid cell size）：
- eps/cell_size < 1 时，数值差分被 hash grid 离散化误差主导
- 相邻采样点落在同一个 cell，SDF 值几乎相同
- 梯度趋近于 0，导致 Eikonal loss 趋近于 1

建议：
1. eps 应该 >= 1 个 cell size，即 eps >= {min_cell_size_world:.6f}
2. 当前 auto_eps = {auto_eps:.6f}，约等于 {auto_eps/min_cell_size_world:.2f} 个 cell
3. eps=1e-6 只有 {1e-6/min_cell_size_world:.4f} 个 cell，太小了！

修复方案：
- 数值梯度: 使用 auto eps（当前实现）
- 解析梯度: 使用 autograd（不受 eps 影响）
""")

    # 额外测试：band 采样对点数的影响
    print("\n" + "="*60)
    print("Band 采样对采样点数的影响")
    print("="*60)

    sdf_values = model.query_vert_sdf(model.active_internal_vertices)
    inv_s = model.backbone.inv_s.item()

    for band in [0.05, 0.1, 0.2, 0.5, 1.0]:
        mask = sdf_values.abs() < band
        n_in_band = mask.sum().item()
        print(f"  band={band:.2f}: {n_in_band} points ({n_in_band/len(sdf_values)*100:.1f}%)")

    # 检查 SDF 分布
    print(f"\n  SDF 分布:")
    print(f"    min={sdf_values.min():.4f}, max={sdf_values.max():.4f}")
    print(f"    mean={sdf_values.mean():.4f}, std={sdf_values.std():.4f}")
    print(f"    inv_s={inv_s:.4f}")


if __name__ == "__main__":
    main()
