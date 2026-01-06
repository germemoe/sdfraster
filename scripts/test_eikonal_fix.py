#!/usr/bin/env python3
"""
测试 Eikonal Loss 修复是否正确。

使用几何初始化时，初始 Eikonal Loss 应该接近 0（因为初始 SDF 是球体，梯度范数约为 1）。
如果 Eikonal Loss 很大（如 0.4-0.8），说明坐标变换或 SDF 缩放有问题。
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import iNGPDW, Model
from utils.model_util import eikonal_loss


def test_geometric_init_gradient():
    """测试几何初始化后的梯度范数"""
    print("="*60)
    print("测试 Eikonal Loss 修复")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建模型参数
    scene_scaling = torch.tensor(2.5, device=device)
    center = torch.zeros(1, 3, device=device)

    # 创建一些随机顶点
    n_verts = 1000
    vertices = torch.randn(n_verts, 3, device=device) * 0.5  # 在 [-0.5, 0.5] 范围内
    ext_vertices = torch.randn(100, 3, device=device) * 2  # 外部点

    # 测试不同配置
    configs = [
        {"use_geometric_init": True, "use_contraction": False},
        {"use_geometric_init": False, "use_contraction": False},
    ]

    for config in configs:
        print(f"\n配置: {config}")

        # 创建模型
        model = Model(
            vertices=vertices.clone(),
            ext_vertices=ext_vertices.clone(),
            center=center.clone(),
            scene_scaling=float(scene_scaling),
            max_sh_deg=3,
            L=8,
            hidden_dim=64,
            **config
        )
        model.eval()

        # 在原点附近采样一些测试点
        test_points = torch.randn(500, 3, device=device) * 0.3

        # 计算 SDF 和梯度
        with torch.no_grad():
            sdf, gradients, hessian = model.compute_sdf_gradients(
                points=test_points,
                mode="numerical",
                taps=6,
                eps=0.01
            )

        # 计算梯度范数
        grad_norms = gradients.norm(dim=-1)
        eik_loss = ((grad_norms - 1) ** 2).mean()

        print(f"  SDF 范围: [{sdf.min():.4f}, {sdf.max():.4f}]")
        print(f"  梯度范数: mean={grad_norms.mean():.4f}, std={grad_norms.std():.4f}")
        print(f"  梯度范数范围: [{grad_norms.min():.4f}, {grad_norms.max():.4f}]")
        print(f"  Eikonal Loss: {eik_loss:.6f}")

        # 检查是否正确
        if config["use_geometric_init"]:
            if eik_loss < 0.1:
                print(f"  ✓ 几何初始化正确！Eikonal Loss 接近 0")
            else:
                print(f"  ✗ 几何初始化有问题！Eikonal Loss 应该接近 0，但得到 {eik_loss:.4f}")

        del model
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    test_geometric_init_gradient()
