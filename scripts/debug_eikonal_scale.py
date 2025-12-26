#!/usr/bin/env python3
"""
诊断 Eikonal Loss 尺度问题的脚本

主要检查:
1. 坐标变换链是否正确
2. eps 计算是否与 Neuralangelo 对齐
3. SDF 输出空间 vs 梯度计算空间是否匹配
4. 几何初始化后的梯度范数
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/data/jk/project/radmesh')

from models.ingp_color import Model, iNGPDW


def test_coordinate_chain():
    """测试坐标变换链"""
    print("\n" + "="*60)
    print("[1] 坐标变换链测试")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建简单模型
    n_points = 100
    vertices = torch.randn(n_points, 3) * 0.5
    ext_vertices = torch.randn(10, 3) * 2
    center = torch.zeros(3, device=device)
    scaling = torch.tensor(2.0)
    sdf_bias = 0.4

    model = Model(
        vertices.to(device),
        ext_vertices,
        center,
        scaling,
        sdf_bias=sdf_bias,
        use_contraction=False
    )

    # 测试点：世界坐标
    test_point_world = torch.tensor([[0.5, 0.0, 0.0]], device=device)

    print(f"\n测试点 (世界坐标): {test_point_world.tolist()}")
    print(f"center: {model.center.tolist()}")
    print(f"scene_scaling: {float(model.scene_scaling)}")

    # 手动追踪坐标变换
    normalized = (test_point_world - model.center) / model.scene_scaling
    print(f"\n1. normalized = (world - center) / scaling")
    print(f"   normalized: {normalized.tolist()}")
    print(f"   范围应该在 [-1, 1]")

    x = (normalized + 1) / 2
    print(f"\n2. x = (normalized + 1) / 2  [hash grid 输入]")
    print(f"   x: {x.tolist()}")
    print(f"   范围应该在 [0, 1]")

    xyz_input = x * 2 - 1
    print(f"\n3. xyz_input = x * 2 - 1  [SDF 网络输入]")
    print(f"   xyz_input: {xyz_input.tolist()}")
    print(f"   xyz_input == normalized? {torch.allclose(xyz_input, normalized)}")

    # 理想球体 SDF
    ideal_sdf_normalized = xyz_input.norm(dim=-1) - sdf_bias
    ideal_sdf_world = ideal_sdf_normalized * float(model.scene_scaling)
    print(f"\n4. 理想球体 SDF:")
    print(f"   |xyz_input| = {xyz_input.norm(dim=-1).item():.4f}")
    print(f"   sdf_normalized = |xyz_input| - sdf_bias = {ideal_sdf_normalized.item():.4f}")
    print(f"   sdf_world = sdf_normalized * scaling = {ideal_sdf_world.item():.4f}")

    # 实际查询
    actual_sdf = model.query_vert_sdf(test_point_world)
    print(f"\n5. 实际 SDF 查询结果: {actual_sdf.item():.4f}")
    print(f"   差异: {abs(actual_sdf.item() - ideal_sdf_world.item()):.4f}")


def test_eps_calculation():
    """测试 eps 计算"""
    print("\n" + "="*60)
    print("[2] eps 计算测试")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    vertices = torch.randn(100, 3) * 0.5
    ext_vertices = torch.randn(10, 3) * 2
    center = torch.zeros(3, device=device)
    scaling = torch.tensor(2.38)  # 用户的实际 scaling

    model = Model(
        vertices.to(device),
        ext_vertices,
        center,
        scaling,
        sdf_bias=0.4687,  # 用户的实际 sdf_bias
        use_contraction=False,
        base_resolution=64,
        per_level_scale=2,
        L=8
    )

    backbone = model.backbone

    print(f"\nHash Grid 配置:")
    print(f"  base_resolution: {backbone.base_resolution}")
    print(f"  per_level_scale: {backbone.per_level_scale}")
    print(f"  L (levels): {backbone.L}")
    print(f"  scene_scaling: {float(model.scene_scaling)}")

    # 计算各级分辨率
    print(f"\n各级分辨率:")
    for i in range(backbone.L):
        res = backbone.base_resolution * (backbone.per_level_scale ** i)
        cell_size_01 = 1.0 / res  # [0,1] 空间
        cell_size_norm = 2.0 / res  # [-1,1] 归一化空间
        cell_size_world = 2.0 * float(model.scene_scaling) / res  # 世界空间
        print(f"  Level {i}: res={res:.0f}, cell_01={cell_size_01:.6f}, cell_norm={cell_size_norm:.6f}, cell_world={cell_size_world:.6f}")

    # compute_sdf_gradients 中的 eps 计算
    finest_resolution = backbone.base_resolution * (backbone.per_level_scale ** (backbone.L - 1))
    eps_current = float(model.scene_scaling) / finest_resolution

    print(f"\n当前 eps 计算 (compute_sdf_gradients):")
    print(f"  finest_resolution = {finest_resolution}")
    print(f"  eps = scene_scaling / finest_resolution = {eps_current:.6f}")

    # eps 在各个空间的含义
    eps_normalized = eps_current / float(model.scene_scaling)
    eps_01 = eps_normalized / 2

    print(f"\neps 在各空间的值:")
    print(f"  世界空间: {eps_current:.6f}")
    print(f"  归一化空间 [-1,1]: {eps_normalized:.6f}")
    print(f"  Hash 空间 [0,1]: {eps_01:.6f}")
    print(f"  最细层 cell size (hash空间): {1.0/finest_resolution:.6f}")
    print(f"  eps / cell_size = {eps_01 / (1.0/finest_resolution):.2f}  (应该 ≈ 1)")

    # Neuralangelo 风格 (供参考)
    print(f"\n[参考] Neuralangelo 风格:")
    print(f"  Neuralangelo 使用 eps = 1/resolution 直接在 [-2,2] 空间")
    print(f"  如果范围是 [-2,2]，cell size = 4/resolution")
    print(f"  eps / cell_size = 1/4 = 0.25")


def test_gradient_math():
    """测试梯度计算的数学"""
    print("\n" + "="*60)
    print("[3] 梯度数学验证")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    vertices = torch.randn(100, 3) * 0.5
    ext_vertices = torch.randn(10, 3) * 2
    center = torch.zeros(3, device=device)
    scaling = torch.tensor(2.0)
    sdf_bias = 0.4

    model = Model(
        vertices.to(device),
        ext_vertices,
        center,
        scaling,
        sdf_bias=sdf_bias,
        use_contraction=False
    )

    # 在球面上采样测试点
    n_samples = 1000
    phi = torch.rand(n_samples, device=device) * 2 * np.pi
    cos_theta = torch.rand(n_samples, device=device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)

    # 球面半径 = sdf_bias (在 xyz_input 空间)
    sphere_radius_norm = sdf_bias
    sphere_radius_world = sphere_radius_norm * float(scaling)

    sphere_dirs = torch.stack([
        sin_theta * torch.cos(phi),
        sin_theta * torch.sin(phi),
        cos_theta
    ], dim=-1)

    # 在球面上的点 (世界坐标)
    sphere_points = sphere_dirs * sphere_radius_world + center

    print(f"\n球面采样点:")
    print(f"  sphere_radius_world: {sphere_radius_world:.4f}")
    print(f"  采样点数: {n_samples}")

    # 理想梯度
    # 对于 sdf(x) = |x - center| - r，梯度 = (x - center) / |x - center|
    ideal_gradients = sphere_dirs  # 归一化方向，模长 = 1

    # 使用 autograd 计算梯度
    sphere_points_ag = sphere_points.clone().requires_grad_(True)
    sdf_ag = model.query_vert_sdf(sphere_points_ag)
    grad_ag = torch.autograd.grad(sdf_ag.sum(), sphere_points_ag)[0]

    print(f"\n[Autograd] 梯度统计:")
    grad_ag_norms = grad_ag.norm(dim=-1)
    print(f"  梯度范数: mean={grad_ag_norms.mean():.4f}, std={grad_ag_norms.std():.4f}")
    print(f"  Eikonal loss: {((grad_ag_norms - 1)**2).mean():.6f}")

    # 梯度方向对齐度
    dot_product = (grad_ag / (grad_ag.norm(dim=-1, keepdim=True) + 1e-8) * ideal_gradients).sum(dim=-1)
    print(f"  方向对齐度: mean={dot_product.mean():.4f} (理想=1)")

    # 使用数值梯度
    _, grad_num, _ = model.compute_sdf_gradients(sphere_points, mode="numerical", taps=6)

    print(f"\n[Numerical 6-tap] 梯度统计:")
    grad_num_norms = grad_num.norm(dim=-1)
    print(f"  梯度范数: mean={grad_num_norms.mean():.4f}, std={grad_num_norms.std():.4f}")
    print(f"  Eikonal loss: {((grad_num_norms - 1)**2).mean():.6f}")

    # 对比
    print(f"\n[对比] Autograd vs Numerical:")
    diff = (grad_ag - grad_num).norm(dim=-1)
    print(f"  梯度差异: mean={diff.mean():.6f}, max={diff.max():.6f}")


def test_ideal_sphere_gradient():
    """测试理想球体 SDF 的梯度（不经过网络）"""
    print("\n" + "="*60)
    print("[4] 理想球体 SDF 梯度测试 (验证数值方法)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    center = torch.zeros(3, device=device)
    radius = 1.0

    def ideal_sphere_sdf(points):
        return (points - center).norm(dim=-1) - radius

    # 采样点
    n_samples = 1000
    phi = torch.rand(n_samples, device=device) * 2 * np.pi
    cos_theta = torch.rand(n_samples, device=device) * 2 - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)
    r = torch.rand(n_samples, device=device) * 2  # [0, 2] 范围

    points = torch.stack([
        r * sin_theta * torch.cos(phi),
        r * sin_theta * torch.sin(phi),
        r * cos_theta
    ], dim=-1) + center

    # Autograd 梯度
    points_ag = points.clone().requires_grad_(True)
    sdf_ag = ideal_sphere_sdf(points_ag)
    grad_ag = torch.autograd.grad(sdf_ag.sum(), points_ag)[0]

    # 数值梯度
    eps = 0.001
    eps_x = torch.tensor([eps, 0., 0.], device=device)
    eps_y = torch.tensor([0., eps, 0.], device=device)
    eps_z = torch.tensor([0., 0., eps], device=device)

    grad_x = (ideal_sphere_sdf(points + eps_x) - ideal_sphere_sdf(points - eps_x)) / (2 * eps)
    grad_y = (ideal_sphere_sdf(points + eps_y) - ideal_sphere_sdf(points - eps_y)) / (2 * eps)
    grad_z = (ideal_sphere_sdf(points + eps_z) - ideal_sphere_sdf(points - eps_z)) / (2 * eps)
    grad_num = torch.stack([grad_x, grad_y, grad_z], dim=-1)

    print(f"\n理想球体 SDF: f(x) = |x - center| - {radius}")
    print(f"采样点数: {n_samples}")

    print(f"\n[Autograd] 梯度范数: mean={grad_ag.norm(dim=-1).mean():.6f}")
    print(f"[Numerical] 梯度范数: mean={grad_num.norm(dim=-1).mean():.6f}")
    print(f"差异: {(grad_ag - grad_num).norm(dim=-1).mean():.8f}")

    print(f"\n结论: 数值方法对理想 SDF 是准确的")


def test_softplus_effect():
    """测试 Softplus 对梯度的影响"""
    print("\n" + "="*60)
    print("[5] Softplus 对梯度的影响")
    print("="*60)

    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 64

    # 简化的 SDF 网络 (只有 xyz 输入)
    class SimpleSDF(nn.Module):
        def __init__(self, activ='softplus'):
            super().__init__()
            self.linear1 = nn.Linear(3, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 1)

            if activ == 'softplus':
                self.activ = nn.Softplus(beta=100)
            else:
                self.activ = nn.ReLU()

            # 几何初始化
            with torch.no_grad():
                nn.init.normal_(self.linear1.weight, 0.0, np.sqrt(2 / hidden_dim))
                nn.init.constant_(self.linear1.bias, 0.0)

                nn.init.normal_(self.linear2.weight, mean=np.sqrt(np.pi / hidden_dim), std=0.0001)
                nn.init.constant_(self.linear2.bias, -0.5)  # sdf_bias = 0.5

        def forward(self, x):
            h = self.activ(self.linear1(x))
            return self.linear2(h)

    # 测试 Softplus vs ReLU
    for activ_name in ['softplus', 'relu']:
        print(f"\n--- {activ_name.upper()} ---")
        net = SimpleSDF(activ=activ_name).to(device)

        # 采样点 (在 [-1, 1] 归一化空间)
        n_samples = 1000
        phi = torch.rand(n_samples, device=device) * 2 * np.pi
        cos_theta = torch.rand(n_samples, device=device) * 2 - 1
        sin_theta = torch.sqrt(1 - cos_theta**2)
        r = torch.rand(n_samples, device=device) * 0.8  # 球内采样

        points = torch.stack([
            r * sin_theta * torch.cos(phi),
            r * sin_theta * torch.sin(phi),
            r * cos_theta
        ], dim=-1)

        points.requires_grad_(True)
        sdf = net(points)
        grad = torch.autograd.grad(sdf.sum(), points)[0]

        grad_norms = grad.norm(dim=-1)
        print(f"  梯度范数: mean={grad_norms.mean():.4f}, std={grad_norms.std():.4f}")
        print(f"  Eikonal loss: {((grad_norms - 1)**2).mean():.6f}")

        # 中心点梯度
        center_point = torch.zeros(1, 3, device=device, requires_grad=True)
        center_sdf = net(center_point)
        center_grad = torch.autograd.grad(center_sdf.sum(), center_point)[0]
        print(f"  中心点 SDF: {center_sdf.item():.4f} (理想: -0.5)")
        print(f"  中心点梯度范数: {center_grad.norm().item():.4f}")


def test_full_model_gradient_distribution():
    """测试完整模型的梯度分布"""
    print("\n" + "="*60)
    print("[6] 完整模型梯度分布")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用和用户类似的配置
    vertices = torch.randn(1000, 3) * 0.5
    ext_vertices = torch.randn(100, 3) * 2
    center = vertices.mean(dim=0).to(device)
    scaling = torch.tensor(2.38)
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

    print(f"\n模型配置:")
    print(f"  scene_scaling: {float(model.scene_scaling):.4f}")
    print(f"  sdf_bias: {sdf_bias}")
    print(f"  center: {model.center.tolist()}")

    # 不同距离的采样
    radii = [0.2, 0.5, 0.8, 1.0, 1.5]  # 相对于 sdf_bias

    for r_factor in radii:
        r = sdf_bias * r_factor
        n_samples = 500

        # 球面采样 (归一化空间)
        phi = torch.rand(n_samples, device=device) * 2 * np.pi
        cos_theta = torch.rand(n_samples, device=device) * 2 - 1
        sin_theta = torch.sqrt(1 - cos_theta**2)

        sphere_dirs = torch.stack([
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            cos_theta
        ], dim=-1)

        # 转换到世界坐标
        points_world = sphere_dirs * r * float(scaling) + center

        # 计算梯度
        _, gradients, _ = model.compute_sdf_gradients(points_world, mode="numerical", taps=6)
        grad_norms = gradients.norm(dim=-1)

        # 查询 SDF
        sdf_values = model.query_vert_sdf(points_world)

        print(f"\n  半径 = {r_factor:.1f} * sdf_bias = {r:.4f} (归一化空间)")
        print(f"    SDF: mean={sdf_values.mean():.4f}, std={sdf_values.std():.4f}")
        print(f"    梯度范数: mean={grad_norms.mean():.4f}, std={grad_norms.std():.4f}")
        print(f"    Eikonal: {((grad_norms - 1)**2).mean():.6f}")


def main():
    print("="*60)
    print("Eikonal Loss 尺度问题诊断")
    print("="*60)

    test_coordinate_chain()
    test_eps_calculation()
    test_ideal_sphere_gradient()
    test_softplus_effect()
    test_gradient_math()
    test_full_model_gradient_distribution()

    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    print("""
关键发现:

1. 坐标变换链:
   - world → normalized → hash[0,1] → xyz_input[-1,1]
   - use_contraction=False 时: xyz_input == normalized

2. SDF 输出:
   - SDF 网络输出在 xyz_input 空间
   - query_vert_sdf 乘以 scene_scaling 转换到世界空间
   - 这样 ∇_world(sdf_world) = ∇_normalized(sdf_normalized)

3. eps 计算:
   - 当前: eps = scene_scaling / finest_resolution
   - 这对应 hash 空间约 0.5 个 cell
   - Neuralangelo 使用约 0.25 个 cell (1/resolution 在 [-2,2] 空间)
   - 差异不大，不是主要问题

4. Softplus 的影响 (主要问题!):
   - Softplus(0) ≈ 0.007 ≠ 0
   - 导致几何初始化的球体 SDF 不精确
   - 梯度在中心附近约为 0.5 (sigmoid(0) = 0.5)
   - 这解释了初始 Eikonal loss ≈ 0.25-0.5

5. 分裂后 Eikonal loss 增大:
   - 可能原因: 新顶点在 hash grid 训练不充分的区域
   - 需要进一步调查分裂逻辑
""")


if __name__ == "__main__":
    main()
