#!/usr/bin/env python3
"""
计算 checkpoint 的 Eikonal Loss

用法:
    python scripts/eval_eikonal_loss.py --ckpt_path output/sdf/DTU/test/xxx
    python scripts/eval_eikonal_loss.py --ckpt_path output/sdf/DTU/test/xxx --band_threshold 0.2
"""
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ingp_color import Model
from models.frozen import FrozenTetModel
from utils.model_util import eikonal_loss, curvature_loss


def compute_eikonal_stats(model, band_threshold=None, eps=None, num_samples=None):
    """
    计算模型的 Eikonal loss 统计信息

    Args:
        model: Model 或 FrozenTetModel
        band_threshold: SDF band 阈值，只计算 |SDF| < threshold 的顶点
        eps: 数值梯度步长，None 则自动计算
        num_samples: 采样点数，None 则使用全部

    Returns:
        dict: 包含各种统计信息
    """
    device = model.device

    # 获取活跃的内部顶点
    if hasattr(model, 'active_internal_vertices'):
        active_verts = model.active_internal_vertices
    else:
        # FrozenTetModel 没有这个属性，使用 interior_vertices
        active_verts = model.interior_vertices

    print(f"[INFO] 活跃内部顶点数: {active_verts.shape[0]}")

    with torch.no_grad():
        # 查询 SDF
        active_sdf = model.query_vert_sdf(active_verts)
        sdf_abs = active_sdf.abs()

        # 采样策略
        if band_threshold is not None:
            mask = sdf_abs < band_threshold
            print(f"[INFO] Band threshold: {band_threshold}, 符合条件的顶点: {mask.sum().item()}")
        elif num_samples is not None and num_samples < active_verts.shape[0]:
            keep_ratio = num_samples / active_verts.shape[0]
            threshold = sdf_abs.quantile(keep_ratio)
            mask = sdf_abs <= threshold
            print(f"[INFO] Quantile sampling: {num_samples} samples, threshold: {threshold:.6f}")
        else:
            mask = torch.ones(active_verts.shape[0], dtype=torch.bool, device=device)
            print(f"[INFO] 使用全部顶点")

        sampled_points = active_verts[mask]
        sampled_sdf = active_sdf[mask]

        if sampled_points.shape[0] == 0:
            print("[WARNING] 没有符合条件的采样点!")
            return None

        print(f"[INFO] 采样点数: {sampled_points.shape[0]}")

        # 计算 SDF 梯度
        _, gradients, hessian_diag = model.compute_sdf_gradients(
            points=sampled_points,
            mode="numerical",
            taps=6,
            eps=eps
        )

        # 计算梯度模长
        grad_norm = torch.norm(gradients, dim=-1)

        # Eikonal loss: (||∇SDF|| - 1)^2
        eik_loss_per_point = (grad_norm - 1).pow(2)
        eik_loss_mean = eik_loss_per_point.mean().item()

        # Curvature loss
        if hessian_diag is not None:
            laplacian = hessian_diag.sum(dim=-1)
            curv_loss_mean = laplacian.abs().mean().item()
        else:
            curv_loss_mean = None

        # 统计信息
        stats = {
            'num_samples': sampled_points.shape[0],
            'eikonal_loss': eik_loss_mean,
            'curvature_loss': curv_loss_mean,
            'grad_norm_mean': grad_norm.mean().item(),
            'grad_norm_std': grad_norm.std().item(),
            'grad_norm_min': grad_norm.min().item(),
            'grad_norm_max': grad_norm.max().item(),
            'sdf_mean': sampled_sdf.mean().item(),
            'sdf_std': sampled_sdf.std().item(),
            'sdf_min': sampled_sdf.min().item(),
            'sdf_max': sampled_sdf.max().item(),
        }

        # 分位数
        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            stats[f'grad_norm_q{int(q*100):02d}'] = grad_norm.quantile(q).item()

        return stats


def main():
    parser = argparse.ArgumentParser(description='计算 checkpoint 的 Eikonal Loss')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Checkpoint 目录路径')
    parser.add_argument('--band_threshold', type=float, default=None,
                        help='SDF band 阈值，只计算 |SDF| < threshold 的顶点')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='采样点数，None 则使用全部')
    parser.add_argument('--eps', type=float, default=None,
                        help='数值梯度步长，None 则自动计算')
    parser.add_argument('--frozen', action='store_true',
                        help='加载 FrozenTetModel 而不是 Model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.ckpt_path)

    print(f"[INFO] 加载 checkpoint: {ckpt_path}")
    print(f"[INFO] Device: {device}")

    # 加载模型
    if args.frozen:
        model = FrozenTetModel.load_ckpt(ckpt_path, device)
    else:
        model = Model.load_ckpt(ckpt_path, device)

    model.eval()

    print(f"[INFO] 模型加载成功")
    print(f"[INFO] 顶点数: {model.vertices.shape[0]}")
    print(f"[INFO] 四面体数: {model.indices.shape[0]}")
    print(f"[INFO] Scene scaling: {model.scene_scaling}")

    # 计算 Eikonal loss
    stats = compute_eikonal_stats(
        model,
        band_threshold=args.band_threshold,
        eps=args.eps,
        num_samples=args.num_samples
    )

    if stats is None:
        return

    # 打印结果
    print("\n" + "=" * 60)
    print("Eikonal Loss 统计")
    print("=" * 60)
    print(f"采样点数:        {stats['num_samples']}")
    print(f"Eikonal Loss:    {stats['eikonal_loss']:.6f}")
    if stats['curvature_loss'] is not None:
        print(f"Curvature Loss:  {stats['curvature_loss']:.6f}")
    print()
    print("梯度模长统计:")
    print(f"  Mean:   {stats['grad_norm_mean']:.4f}")
    print(f"  Std:    {stats['grad_norm_std']:.4f}")
    print(f"  Min:    {stats['grad_norm_min']:.4f}")
    print(f"  Max:    {stats['grad_norm_max']:.4f}")
    print(f"  Median: {stats['grad_norm_q50']:.4f}")
    print()
    print("梯度模长分位数:")
    print(f"  1%:    {stats['grad_norm_q01']:.4f}")
    print(f"  5%:    {stats['grad_norm_q05']:.4f}")
    print(f"  25%:   {stats['grad_norm_q25']:.4f}")
    print(f"  75%:   {stats['grad_norm_q75']:.4f}")
    print(f"  95%:   {stats['grad_norm_q95']:.4f}")
    print(f"  99%:   {stats['grad_norm_q99']:.4f}")
    print()
    print("SDF 值统计:")
    print(f"  Mean:  {stats['sdf_mean']:.6f}")
    print(f"  Std:   {stats['sdf_std']:.6f}")
    print(f"  Min:   {stats['sdf_min']:.6f}")
    print(f"  Max:   {stats['sdf_max']:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
