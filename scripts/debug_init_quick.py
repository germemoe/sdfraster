#!/usr/bin/env python3
"""快速测试几何初始化的问题"""
import torch
import torch.nn as nn
import numpy as np

def test_geometric_init():
    """测试几何初始化是否正确"""
    print("="*60)
    print("几何初始化测试")
    print("="*60)

    hidden_dim = 64
    sdf_bias = 0.5

    # 模拟 radmesh 的 SDF 网络（只有 xyz 输入）
    class SimpleSDF(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 1)
            self.activ = nn.Softplus(beta=100)

            # 几何初始化
            with torch.no_grad():
                nn.init.normal_(self.linear1.weight, 0.0, np.sqrt(2 / hidden_dim))
                nn.init.constant_(self.linear1.bias, 0.0)
                nn.init.normal_(self.linear2.weight, mean=np.sqrt(np.pi / hidden_dim), std=0.0001)
                nn.init.constant_(self.linear2.bias, -sdf_bias)

        def forward(self, x):
            return self.linear2(self.activ(self.linear1(x)))

    net = SimpleSDF()

    # 测试点
    # 1. 中心点：理想 SDF = -sdf_bias
    center = torch.zeros(1, 3)
    # 2. 球面点：理想 SDF = 0
    sphere_pts = torch.randn(100, 3)
    sphere_pts = sphere_pts / sphere_pts.norm(dim=-1, keepdim=True) * sdf_bias
    # 3. 外部点：理想 SDF > 0
    outer_pts = torch.randn(100, 3)
    outer_pts = outer_pts / outer_pts.norm(dim=-1, keepdim=True) * (sdf_bias * 1.5)

    print(f"\nsdf_bias = {sdf_bias}")
    print(f"\n中心点 (理想 SDF = -{sdf_bias}):")
    print(f"  实际 SDF = {net(center).item():.4f}")

    print(f"\n球面点 |x| = {sdf_bias} (理想 SDF = 0):")
    sphere_sdf = net(sphere_pts)
    print(f"  实际 SDF: mean={sphere_sdf.mean():.4f}, std={sphere_sdf.std():.4f}")

    print(f"\n外部点 |x| = {sdf_bias * 1.5} (理想 SDF = {sdf_bias * 0.5}):")
    outer_sdf = net(outer_pts)
    print(f"  实际 SDF: mean={outer_sdf.mean():.4f}, std={outer_sdf.std():.4f}")

    # 测试梯度
    print(f"\n梯度测试:")
    test_pts = torch.randn(1000, 3) * sdf_bias
    test_pts.requires_grad_(True)
    sdf = net(test_pts)
    grad = torch.autograd.grad(sdf.sum(), test_pts)[0]
    grad_norms = grad.norm(dim=-1)
    eikonal = ((grad_norms - 1)**2).mean()

    print(f"  梯度范数: mean={grad_norms.mean():.4f}, std={grad_norms.std():.4f}")
    print(f"  Eikonal loss = {eikonal:.4f}")

    # 对比 ReLU
    print("\n" + "="*60)
    print("对比: 使用 ReLU 代替 Softplus")
    print("="*60)

    class SimpleSDF_ReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(3, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 1)
            self.activ = nn.ReLU()

            with torch.no_grad():
                nn.init.normal_(self.linear1.weight, 0.0, np.sqrt(2 / hidden_dim))
                nn.init.constant_(self.linear1.bias, 0.0)
                nn.init.normal_(self.linear2.weight, mean=np.sqrt(np.pi / hidden_dim), std=0.0001)
                nn.init.constant_(self.linear2.bias, -sdf_bias)

        def forward(self, x):
            return self.linear2(self.activ(self.linear1(x)))

    net_relu = SimpleSDF_ReLU()

    print(f"\n中心点 (理想 SDF = -{sdf_bias}):")
    print(f"  Softplus: {net(center).item():.4f}")
    print(f"  ReLU:     {net_relu(center).item():.4f}")

    sphere_sdf_relu = net_relu(sphere_pts)
    print(f"\n球面点 (理想 SDF = 0):")
    print(f"  Softplus: mean={sphere_sdf.mean():.4f}")
    print(f"  ReLU:     mean={sphere_sdf_relu.mean():.4f}")

    test_pts2 = torch.randn(1000, 3) * sdf_bias
    test_pts2.requires_grad_(True)
    sdf2 = net_relu(test_pts2)
    grad2 = torch.autograd.grad(sdf2.sum(), test_pts2)[0]
    grad_norms2 = grad2.norm(dim=-1)
    eikonal2 = ((grad_norms2 - 1)**2).mean()

    print(f"\n梯度范数:")
    print(f"  Softplus: mean={grad_norms.mean():.4f}, Eikonal={eikonal:.4f}")
    print(f"  ReLU:     mean={grad_norms2.mean():.4f}, Eikonal={eikonal2:.4f}")


def test_hash_features_impact():
    """测试 hash features 对几何初始化的影响"""
    print("\n" + "="*60)
    print("Hash Features 对几何初始化的影响")
    print("="*60)

    hidden_dim = 64
    sdf_bias = 0.5
    hash_dim = 32  # 典型值: 8 levels * 4 features

    class SDFWithHash(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = 3 + hash_dim
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 1)
            self.activ = nn.Softplus(beta=100)

            # 几何初始化
            with torch.no_grad():
                nn.init.normal_(self.linear1.weight, 0.0, np.sqrt(2 / hidden_dim))
                nn.init.constant_(self.linear1.bias, 0.0)
                # hash 部分权重设为 0
                self.linear1.weight[:, 3:].zero_()

                nn.init.normal_(self.linear2.weight, mean=np.sqrt(np.pi / hidden_dim), std=0.0001)
                nn.init.constant_(self.linear2.bias, -sdf_bias)

        def forward(self, xyz, hash_feat):
            x = torch.cat([xyz, hash_feat], dim=-1)
            return self.linear2(self.activ(self.linear1(x)))

    net = SDFWithHash()

    # 测试：hash features 为 0 时
    test_pts = torch.randn(100, 3) * sdf_bias
    zero_hash = torch.zeros(100, hash_dim)

    print(f"\nHash features = 0:")
    sdf = net(test_pts, zero_hash)
    print(f"  SDF: mean={sdf.mean():.4f}, std={sdf.std():.4f}")

    # 测试：hash features 为随机值时（模拟训练后）
    rand_hash = torch.randn(100, hash_dim) * 0.1  # 小随机值
    sdf_rand = net(test_pts, rand_hash)
    print(f"\nHash features = random (std=0.1):")
    print(f"  SDF: mean={sdf_rand.mean():.4f}, std={sdf_rand.std():.4f}")

    # 关键：即使初始化时 hash 权重为 0，第一层输出也会受影响
    print(f"\n第一层权重分析:")
    print(f"  xyz 部分: norm={net.linear1.weight[:, :3].norm():.4f}")
    print(f"  hash 部分: norm={net.linear1.weight[:, 3:].norm():.4f}")

    # 训练几步后
    print(f"\n模拟训练后 (hash 权重变为随机):")
    with torch.no_grad():
        net.linear1.weight[:, 3:].normal_(0, 0.01)
    print(f"  hash 部分: norm={net.linear1.weight[:, 3:].norm():.4f}")

    sdf_after = net(test_pts, rand_hash)
    print(f"  SDF: mean={sdf_after.mean():.4f}, std={sdf_after.std():.4f}")
    print(f"  SDF 变化: {(sdf_after - sdf).abs().mean():.4f}")


def test_network_depth():
    """测试网络深度对 Eikonal 的影响"""
    print("\n" + "="*60)
    print("网络深度对 Eikonal 的影响")
    print("="*60)

    sdf_bias = 0.5

    for n_layers, hidden in [(2, 64), (4, 128), (8, 256)]:
        layers = []
        in_dim = 3
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Softplus(beta=100))
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))

        net = nn.Sequential(*layers)

        # 简单初始化（不是完整的几何初始化）
        with torch.no_grad():
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            # 输出层偏置
            net[-1].bias.fill_(-sdf_bias)

        test_pts = torch.randn(1000, 3) * sdf_bias
        test_pts.requires_grad_(True)
        sdf = net(test_pts)
        grad = torch.autograd.grad(sdf.sum(), test_pts)[0]
        grad_norms = grad.norm(dim=-1)
        eikonal = ((grad_norms - 1)**2).mean()

        print(f"\n{n_layers} 层, {hidden} 维:")
        print(f"  梯度范数: mean={grad_norms.mean():.4f}, std={grad_norms.std():.4f}")
        print(f"  Eikonal loss = {eikonal:.4f}")


if __name__ == "__main__":
    test_geometric_init()
    test_hash_features_impact()
    test_network_depth()

    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("""
关键发现:

1. Softplus vs ReLU:
   - Softplus(beta=100) 接近 ReLU，但在 0 附近有平滑
   - 这导致几何初始化不完全精确
   - 但差异应该不大（~0.01 量级）

2. Hash Features 的影响:
   - 初始化时 hash 权重为 0，但训练后会变化
   - hash features 维度 (32) >> xyz 维度 (3)
   - 网络可能过度依赖 hash 而忽略几何结构

3. 网络深度:
   - 2 层太浅，难以表示复杂 SDF
   - Neuralangelo 用 8 层 256 维
   - 更深的网络有更好的表达能力

建议:
1. 增加网络深度 (至少 4-6 层)
2. 增加隐藏层维度 (128-256)
3. 考虑在中间层注入 xyz (参考 Neuralangelo)
4. 或者分离 xyz 和 hash 的处理路径
""")
