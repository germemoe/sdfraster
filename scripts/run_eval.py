#!/usr/bin/env python3
"""
Standalone Mesh Evaluation Script

独立的mesh评估脚本，可以在训练完成后单独运行。

使用方法：
    # 评估训练坐标系的mesh（会自动cull和对齐）
    python scripts/run_eval.py \\
        --mesh output/exp1/final_mesh.ply \\
        --model_path output/exp1 \\
        --dataset_path /path/to/DTU/scan37 \\
        --dtu_path /path/to/DTU \\
        --scan_id 37

    # 评估已经对齐到DTU坐标系的mesh（跳过cull和对齐）
    python scripts/run_eval.py \\
        --mesh output/exp1/mesh_aligned.ply \\
        --dtu_path /path/to/DTU \\
        --scan_id 37 \\
        --already_aligned
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Standalone Mesh Evaluation")
    parser.add_argument('--mesh', type=str, required=True,
                        help='Path to input mesh')
    parser.add_argument('--dtu_path', type=str, required=True,
                        help='Path to DTU dataset root (containing ObsMask/, Points/, Calibration/)')
    parser.add_argument('--scan_id', type=int, required=True,
                        help='DTU scan ID (e.g., 37)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as mesh)')

    # 对于需要cull和对齐的mesh
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model output path (for loading cameras, required if not --already_aligned)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Dataset path (required if not --already_aligned)')

    # 跳过cull和对齐
    parser.add_argument('--already_aligned', action='store_true',
                        help='Skip cull and alignment (mesh is already in DTU world coordinates)')
    parser.add_argument('--no_cull', action='store_true',
                        help='Skip mesh culling')
    parser.add_argument('--dilation_radius', type=int, default=6,
                        help='Mask dilation radius for culling')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.mesh)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.already_aligned:
        # 直接评估已对齐的mesh
        print(f"[EVAL] Evaluating pre-aligned mesh: {args.mesh}")
        from eval.dtu.eval_core import evaluate_mesh
        results = evaluate_mesh(
            mesh_path=args.mesh,
            scan_id=args.scan_id,
            dataset_dir=args.dtu_path,
            vis_out_dir=os.path.join(args.output_dir, 'vis'),
        )
        print(f"\n[EVAL] Results:")
        print(f"    mean_d2s: {results['mean_d2s']:.4f}")
        print(f"    mean_s2d: {results['mean_s2d']:.4f}")
        print(f"    overall:  {results['overall']:.4f}")

    else:
        # 需要加载相机进行cull和对齐
        if args.model_path is None or args.dataset_path is None:
            parser.error("--model_path and --dataset_path are required when not using --already_aligned")

        print(f"[EVAL] Loading cameras from {args.dataset_path}...")

        import torch
        from data import loader

        train_cameras, _, scene_info = loader.load_dataset(
            Path(args.dataset_path),
            'images',
            data_device='cuda',
            eval=True,
            resolution=1.0
        )
        print(f"    Loaded {len(train_cameras)} training cameras")

        # 使用evaluate.py中的完整流程
        from eval.dtu.evaluate import evaluate_dtu_mesh

        results = evaluate_dtu_mesh(
            mesh_path=args.mesh,
            train_cameras=train_cameras,
            dtu_path=args.dtu_path,
            scan_id=args.scan_id,
            output_dir=args.output_dir,
            cull=not args.no_cull,
            dilation_radius=args.dilation_radius,
        )

        if results:
            print(f"\n[EVAL] Results:")
            print(f"    mean_d2s: {results['mean_d2s']:.4f}")
            print(f"    mean_s2d: {results['mean_s2d']:.4f}")
            print(f"    overall:  {results['overall']:.4f}")
        else:
            print("[EVAL] Evaluation failed")
            sys.exit(1)


if __name__ == '__main__':
    main()
