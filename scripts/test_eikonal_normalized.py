#!/usr/bin/env python3
"""
测试：在归一化空间直接计算 Eikonal Loss

目的：排除世界空间转换带来的问题
- 直接在归一化空间 [-1, 1] 采样
- eps = 1/resolution（归一化空间）
- SDF 不乘 scaling
- 对比现有实现（世界空间）的结果
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/data/jk/project/radmesh')

from models.ingp_color import Model


def query_sdf_normalized_space(model, points_normalized):
    """
    直接在归一化空间查询 SDF，不乘 scaling

    Args:
        points_normalized: (N, 3) 归一化坐标 [-1, 1]
    Returns:
        sdf_normalized: (N,) 归一化空间的 SDF
    """
    device = model.device
    points_normalized = points_normalized.to(device)

    sdf_list = []
    chunk_size = model.chunk_size

    for start in range(0, points_normalized.shape[0], chunk_size):
        end = min(start + chunk_size, points_normalized.shape[0])
        pts_norm = points_normalized[start:end]

        # 归一化 → hash [0, 1]
        x = (pts_norm + 1) / 2

        # Hash Grid 编码 + C2F
        h = model.backbone.encoding(x)
        if model.backbone.coarse2fine_enabled and model.backbone.active_levels < model.backbone.L:
            h = h * model.backbone._get_c2f_mask(h)
        h = h.float()

        # SDF 网络输入
        xyz_input = x * 2 - 1  # = pts_norm
        sdf_input = torch.cat([xyz_input, h], dim=-1)

        # SDF 网络输出（不乘 scaling！）
        sdf_normalized = model.backbone.sdf_net(sdf_input)
        sdf_list.append(sdf_normalized.reshape(-1))

    return torch.cat(sdf_list)


def compute_gradient_normalized_space(model, points_normalized, eps):
    """
    在归一化空间用数值差分计算梯度

    Args:
        points_normalized: (N, 3) 归一化坐标
        eps: 归一化空间的步长
    Returns:
        sdf: (N,) SDF 值
        gradients: (N, 3) 梯度
    """
    device = model.device
    points = points_normalized.to(device)

    # 中心点 SDF
    sdf = query_sdf_normalized_space(model, points)

    # 6 个偏移方向
    eps_x = torch.tensor([eps, 0., 0.], device=device)
    eps_y = torch.tensor([0., eps, 0.], device=device)
    eps_z = torch.tensor([0., 0., eps], device=device)

    # 计算 6 个偏移点的 SDF
    sdf_x_pos = query_sdf_normalized_space(model, points + eps_x)
    sdf_x_neg = query_sdf_normalized_space(model, points - eps_x)
    sdf_y_pos = query_sdf_normalized_space(model, points + eps_y)
    sdf_y_neg = query_sdf_normalized_space(model, points - eps_y)
    sdf_z_pos = query_sdf_normalized_space(model, points + eps_z)
    sdf_z_neg = query_sdf_normalized_space(model, points - eps_z)

    # 中心差分
    gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
    gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
    gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
    gradients = torch.stack([gradient_x, gradient_y, gradient_z], dim=-1)

    return sdf, gradients


def main():
    print("=" * 70)
    print("Eikonal Loss 归一化空间 vs 世界空间 对比测试")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建测试模型
    n_points = 1000
    torch.manual_seed(42)
    vertices = torch.randn(n_points, 3) * 0.5
    ext_vertices = torch.randn(100, 3) * 2
    center = vertices.mean(dim=0).to(device)
    scaling = torch.tensor(2.38)
    sdf_bias = 0.4687

    print(f"\n模型配置:")
    print(f"  center: {center.tolist()}")
    print(f"  scaling: {float(scaling)}")
    print(f"  sdf_bias: {sdf_bias}")

    model = Model(
        vertices.to(device),
        ext_vertices,
        center,
        scaling,
        sdf_bias=sdf_bias,
        use_contraction=False,
        base_resolution=64,
        per_level_scale=2,
        L=8,
        hidden_dim=64,
    )

    # 计算 eps
    finest_resolution = model.backbone.base_resolution * (
        model.backbone.per_level_scale ** (model.backbone.L - 1)
    )
    eps_normalized = 1.0 / finest_resolution
    eps_world = float(model.scene_scaling) / finest_resolution

    print(f"\neps 配置:")
    print(f"  finest_resolution: {finest_resolution}")
    print(f"  eps_normalized (归一化空间): {eps_normalized:.6f}")
    print(f"  eps_world (世界空间): {eps_world:.6f}")

    # 采样测试点（在初始化球面附近）
    n_samples = 1000
    torch.manual_seed(123)

    # 在归一化空间采样
    phi = torch.rand(n_samples, device=device) * 2 * np.pi
    cos_theta = torch.rand(n_samples, device=device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)

    sphere_dirs = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta
    ], dim=-1)

    # 不同半径的测试
    test_radii = [0.5 * sdf_bias, sdf_bias, 1.5 * sdf_bias]

    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)

    for r_factor, radius in zip([0.5, 1.0, 1.5], test_radii):
        print(f"\n--- 半径 = {r_factor} × sdf_bias = {radius:.4f} (归一化空间) ---")

        # 归一化空间的点
        points_normalized = sphere_dirs * radius

        # 对应的世界空间点
        points_world = points_normalized * float(scaling) + center

        # ========== 方法1：归一化空间直接计算 ==========
        sdf_norm, grad_norm = compute_gradient_normalized_space(
            model, points_normalized, eps_normalized
        )
        grad_norm_norms = grad_norm.norm(dim=-1)
        eikonal_norm = ((grad_norm_norms - 1) ** 2).mean().item()

        print(f"\n  [归一化空间] eps={eps_normalized:.6f}")
        print(f"    SDF: mean={sdf_norm.mean():.4f}, std={sdf_norm.std():.4f}")
        print(f"    梯度范数: mean={grad_norm_norms.mean():.4f}, std={grad_norm_norms.std():.4f}")
        print(f"    Eikonal loss: {eikonal_norm:.6f}")

        # ========== 方法2：世界空间计算（现有实现） ==========
        sdf_world, grad_world, _ = model.compute_sdf_gradients(
            points_world, mode="numerical", taps=6, eps=eps_world
        )
        grad_world_norms = grad_world.norm(dim=-1)
        eikonal_world = ((grad_world_norms - 1) ** 2).mean().item()

        print(f"\n  [世界空间] eps={eps_world:.6f}")
        print(f"    SDF: mean={sdf_world.mean():.4f}, std={sdf_world.std():.4f}")
        print(f"    梯度范数: mean={grad_world_norms.mean():.4f}, std={grad_world_norms.std():.4f}")
        print(f"    Eikonal loss: {eikonal_world:.6f}")

        # ========== 对比 ==========
        print(f"\n  [对比]")
        print(f"    SDF 比值 (world/norm): {sdf_world.mean() / (sdf_norm.mean() + 1e-8):.4f} (应该≈{float(scaling):.2f})")
        print(f"    梯度范数差异: {abs(grad_norm_norms.mean() - grad_world_norms.mean()):.6f}")
        print(f"    Eikonal 差异: {abs(eikonal_norm - eikonal_world):.6f}")

    # ========== 理想球体对比 ==========
    print("\n" + "=" * 70)
    print("理想球体 SDF 参考值")
    print("=" * 70)

    for r_factor, radius in zip([0.5, 1.0, 1.5], test_radii):
        points_normalized = sphere_dirs * radius

        # 理想球体 SDF = |x| - sdf_bias
        ideal_sdf = points_normalized.norm(dim=-1) - sdf_bias
        # 理想梯度 = x / |x|，范数 = 1
        ideal_grad_norm = 1.0

        # 实际值
        actual_sdf = query_sdf_normalized_space(model, points_normalized)

        print(f"\n  半径 = {r_factor} × sdf_bias:")
        print(f"    理想 SDF: mean={ideal_sdf.mean():.4f}")
        print(f"    实际 SDF: mean={actual_sdf.mean():.4f}")
        print(f"    SDF 误差: {(actual_sdf - ideal_sdf).abs().mean():.4f}")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
如果「归一化空间」和「世界空间」的 Eikonal loss 相近：
  → 坐标变换没问题，问题在几何初始化本身

如果「归一化空间」的 Eikonal loss 明显更好：
  → 坐标变换有问题，需要检查 scaling 的处理

如果两者都很大（远离 0）：
  → 几何初始化本身就不精确（网络容量不足）
""")


if __name__ == "__main__":
    main()
