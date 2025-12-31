#!/usr/bin/env python3
"""
诊断几何初始化问题的脚本
运行方式: python scripts/debug_geometric_init.py
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/data/jk/project/radmesh')

from models.ingp_color import iNGPDW, Model

def check_sdf_net_weights(backbone: iNGPDW):
    """检查 SDF 网络权重是否正确初始化"""
    print("\n" + "="*60)
    print("[1] SDF 网络权重检查")
    print("="*60)

    sdf_net = backbone.sdf_net
    linear0 = sdf_net.linear0
    linear_sdf = sdf_net.linear_sdf

    # 检查第一层
    xyz_weight = linear0.weight[:, :3]
    hash_weight = linear0.weight[:, 3:]

    print(f"\nlinear0 (input → hidden):")
    print(f"  shape: {linear0.weight.shape}")
    print(f"  xyz 部分权重范数:  {xyz_weight.norm():.4f} (应该非零)")
    print(f"  hash 部分权重范数: {hash_weight.norm():.6f} (应该为 0)")
    print(f"  bias 范数: {linear0.bias.norm():.6f} (应该为 0)")

    # 检查 SDF 层
    hidden_dim = linear_sdf.in_features
    expected_mean = np.sqrt(np.pi / hidden_dim)
    print(f"\nlinear_sdf (hidden → 1):")
    print(f"  权重 mean: {linear_sdf.weight.mean():.6f} (理想: {expected_mean:.6f})")
    print(f"  权重 std:  {linear_sdf.weight.std():.6f} (理想: ~0.0001)")
    print(f"  bias: {linear_sdf.bias.item():.4f} (应该为 -sdf_bias)")

    # 诊断
    if hash_weight.norm() > 1e-6:
        print("\n⚠️  警告: hash 部分权重不为零！几何初始化可能失败。")
    else:
        print("\n✓ 权重初始化正确")


def check_coordinate_transform(model: Model, use_contraction: bool):
    """检查坐标变换是否正确"""
    print("\n" + "="*60)
    print("[2] 坐标变换检查")
    print("="*60)

    print(f"\nuse_contraction = {use_contraction}")
    print(f"scene_scaling = {float(model.scene_scaling):.4f}")
    print(f"center = {model.center.tolist()}")
    print(f"sdf_bias = {model.backbone.sdf_bias:.4f}")

    # 在网络输入空间中，球面半径应该等于 sdf_bias
    # 测试球面上的点是否 SDF ≈ 0
    sdf_bias = model.backbone.sdf_bias

    # 生成球面点（在 xyz_input 空间）
    n_samples = 1000
    phi = torch.rand(n_samples, device=model.device) * 2 * np.pi
    cos_theta = torch.rand(n_samples, device=model.device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)
    sphere_dirs = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta
    ], dim=-1)

    # 反推到世界坐标
    if use_contraction:
        # xyz_input = cv/2, 假设 |cv| < 1 时 cv = normalized
        # 所以 xyz_input = normalized/2
        sphere_normalized = sphere_dirs * sdf_bias * 2
    else:
        # xyz_input = normalized
        sphere_normalized = sphere_dirs * sdf_bias

    sphere_world = sphere_normalized * float(model.scene_scaling) + model.center

    # 查询 SDF
    with torch.no_grad():
        sphere_sdf = model.query_vert_sdf(sphere_world)

    print(f"\n球面测试 (半径 = sdf_bias = {sdf_bias:.4f}):")
    print(f"  世界坐标范围: [{sphere_world.min():.4f}, {sphere_world.max():.4f}]")
    print(f"  SDF: mean={sphere_sdf.mean():.4f}, std={sphere_sdf.std():.4f}")
    print(f"  (理想: mean≈0, std≈0)")

    # 测试中心点
    center_sdf = model.query_vert_sdf(model.center.reshape(1, 3))
    print(f"\n中心点 SDF: {center_sdf.item():.4f} (理想: -{sdf_bias:.4f})")


def check_gradient_computation(model: Model):
    """检查梯度计算"""
    print("\n" + "="*60)
    print("[3] 梯度计算检查")
    print("="*60)

    # 在中心附近采样
    n_samples = 1000
    sdf_bias = model.backbone.sdf_bias

    # 生成球内均匀采样点
    phi = torch.rand(n_samples, device=model.device) * 2 * np.pi
    cos_theta = torch.rand(n_samples, device=model.device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)
    r = torch.rand(n_samples, device=model.device) ** (1/3) * sdf_bias * 0.8

    sample_dirs = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta
    ], dim=-1)

    if model.use_contraction:
        sample_normalized = sample_dirs * r.unsqueeze(-1) * 2
    else:
        sample_normalized = sample_dirs * r.unsqueeze(-1)

    sample_world = sample_normalized * float(model.scene_scaling) + model.center

    # 计算梯度
    with torch.no_grad():
        sdf, gradients, _ = model.compute_sdf_gradients(sample_world, mode="numerical", taps=6)

    grad_norms = gradients.norm(dim=-1)
    eikonal_loss = ((grad_norms - 1) ** 2).mean().item()

    print(f"\n采样点数: {n_samples}")
    print(f"梯度范数: mean={grad_norms.mean():.4f}, std={grad_norms.std():.4f}")
    print(f"Eikonal loss: {eikonal_loss:.6f}")
    print(f"(理想: 梯度范数≈1, Eikonal loss≈0)")

    # 检查梯度是否指向径向
    expected_dir = sample_dirs
    actual_dir = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-8)
    dot_product = (expected_dir * actual_dir).sum(dim=-1)
    print(f"\n梯度方向与径向的对齐度: mean={dot_product.mean():.4f}")
    print(f"(理想: ≈1，表示梯度指向外)")


def main():
    print("="*60)
    print("几何初始化诊断")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试模型
    # 模拟点云初始化
    n_points = 1000
    vertices = torch.randn(n_points, 3) * 0.5  # 随机点云
    ext_vertices = torch.randn(100, 3) * 2  # 外部点
    center = vertices.mean(dim=0).to(device)
    scaling = torch.tensor(2.0)

    # 测试两种配置
    for use_contraction in [True, False]:
        print(f"\n\n{'#'*60}")
        print(f"# 测试配置: use_contraction = {use_contraction}")
        print(f"{'#'*60}")

        # 计算正确的 sdf_bias
        pcd_radius = 0.8
        if use_contraction:
            sdf_bias = pcd_radius / float(scaling) / 2
        else:
            sdf_bias = pcd_radius / float(scaling)

        print(f"pcd_radius = {pcd_radius}")
        print(f"sdf_bias = {sdf_bias}")

        model = Model(
            vertices.to(device),
            ext_vertices,
            center,
            scaling,
            sdf_bias=sdf_bias,
            use_contraction=use_contraction
        )

        check_sdf_net_weights(model.backbone)
        check_coordinate_transform(model, use_contraction)
        check_gradient_computation(model)


if __name__ == "__main__":
    main()
