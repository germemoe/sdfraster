"""DTU Mesh Evaluation Wrapper.

直接调用 eval_dtu_vanilla 脚本进行评估（参考 svraster）。
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def evaluate_dtu_mesh(
    input_mesh: str,
    scan_id: str,
    output_dir: str,
    mask_dir: str,
    dtu_dir: str,
):
    """
    评估 DTU 数据集上的 mesh。

    Args:
        input_mesh: 输入 mesh 路径
        scan_id: DTU scan ID（例如 "37"）
        output_dir: 输出目录
        mask_dir: mask 目录（包含 scan{id} 子目录）
        dtu_dir: DTU 官方数据集目录（包含 Points/stl 和 ObsMask）
    """
    script_dir = Path(__file__).parent.parent / 'eval_dtu_vanilla'
    script_path = script_dir / 'evaluate_single_scene.py'

    if not script_path.exists():
        raise FileNotFoundError(f"评估脚本不存在: {script_path}")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(script_path),
        '--input_mesh', str(input_mesh),
        '--scan_id', str(scan_id),
        '--output_dir', str(output_dir),
        '--mask_dir', str(mask_dir),
        '--DTU', str(dtu_dir),
    ]

    print('[EVAL-DTU] Running:', ' '.join(cmd))
    result = subprocess.run(cmd, check=False)

    # 读取结果
    results_path = Path(output_dir) / 'results.json'
    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)
        print(f"[EVAL-DTU] Results:")
        print(f"  Accuracy (d2s):     {results.get('mean_d2s', 'N/A'):.6f}")
        print(f"  Completeness (s2d): {results.get('mean_s2d', 'N/A'):.6f}")
        print(f"  Chamfer Distance:   {results.get('overall', 'N/A'):.6f}")
        return results
    else:
        print(f"[EVAL-DTU] 结果文件不存在: {results_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description='DTU Mesh Evaluation')
    parser.add_argument('--input_mesh', type=str, required=True,
                        help='输入 mesh 路径')
    parser.add_argument('--scan_id', type=str, required=True,
                        help='DTU scan ID（例如 37）')
    parser.add_argument('--output_dir', type=str, default='./eval_output',
                        help='输出目录')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='mask 目录（包含 scan{id} 子目录）')
    parser.add_argument('--dtu_dir', type=str, required=True,
                        help='DTU 官方数据集目录')

    args = parser.parse_args()

    evaluate_dtu_mesh(
        input_mesh=args.input_mesh,
        scan_id=args.scan_id,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        dtu_dir=args.dtu_dir,
    )


if __name__ == '__main__':
    main()
