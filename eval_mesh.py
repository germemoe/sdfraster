#!/usr/bin/env python
"""
DTU Mesh Evaluation Script

独立的 mesh 评测脚本，输入 mesh 路径和 scan_id 即可计算 Chamfer Distance。
参考 svraster 的 eval_dtu_vanilla 实现。

使用方法：
    python eval_mesh.py --input_mesh output/final_mesh.ply --scan_id 37 --dtu_root /path/to/dtu

DTU 数据集目录结构要求：
    dtu_root/
    ├── scan37/                    # 或其他 scan 目录
    │   ├── images/
    │   ├── mask/
    │   └── cameras.npz
    ├── Points/
    │   └── stl/
    │       └── stl037_total.ply   # GT 点云
    └── ObsMask/
        ├── ObsMask37_10.mat
        └── Plane37.mat
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='DTU Mesh Evaluation - 计算 Chamfer Distance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估单个 mesh
    python eval_mesh.py --input_mesh output/final_mesh.ply --scan_id 37 --dtu_root /data/DTU

    # 指定输出目录
    python eval_mesh.py --input_mesh mesh.ply --scan_id 37 --dtu_root /data/DTU --output_dir ./eval_results
        """
    )
    parser.add_argument('--input_mesh', type=str, required=True,
                        help='输入 mesh 路径 (.ply)')
    parser.add_argument('--scan_id', type=str, required=True,
                        help='DTU scan ID（例如 37, 24, 65 等）')
    parser.add_argument('--dtu_root', type=str, required=True,
                        help='DTU 数据集根目录（包含 scan{id}/, Points/, ObsMask/）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认为 mesh 同目录下的 eval_dtu/）')

    args = parser.parse_args()

    # 检查输入文件
    input_mesh = Path(args.input_mesh)
    if not input_mesh.exists():
        print(f"[Error] Mesh 文件不存在: {input_mesh}")
        sys.exit(1)

    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_mesh.parent / 'eval_dtu'

    # 检查 DTU 目录
    dtu_root = Path(args.dtu_root)
    scan_dir = dtu_root / f'scan{args.scan_id}'
    if not scan_dir.exists():
        print(f"[Error] Scan 目录不存在: {scan_dir}")
        sys.exit(1)

    # 查找评估脚本
    script_dir = Path(__file__).parent / 'eval_dtu_vanilla'
    script_path = script_dir / 'evaluate_single_scene.py'
    if not script_path.exists():
        print(f"[Error] 评估脚本不存在: {script_path}")
        sys.exit(1)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = [
        sys.executable, str(script_path),
        '--input_mesh', str(input_mesh),
        '--scan_id', str(args.scan_id),
        '--output_dir', str(output_dir),
        '--mask_dir', str(dtu_root),
        '--DTU', str(dtu_root),
    ]

    print('=' * 60)
    print('[EVAL-DTU] DTU Mesh Evaluation')
    print('=' * 60)
    print(f'  Input mesh:  {input_mesh}')
    print(f'  Scan ID:     {args.scan_id}')
    print(f'  DTU root:    {dtu_root}')
    print(f'  Output dir:  {output_dir}')
    print('=' * 60)
    print('[EVAL-DTU] Running:', ' '.join(cmd))
    print()

    # 运行评估
    result = subprocess.run(cmd, check=False)

    # 读取并显示结果
    results_path = output_dir / 'results.json'
    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)
        print()
        print('=' * 60)
        print('[EVAL-DTU] Results:')
        print('=' * 60)
        print(f"  Accuracy (d2s):     {results.get('mean_d2s', 'N/A'):.6f}")
        print(f"  Completeness (s2d): {results.get('mean_s2d', 'N/A'):.6f}")
        print(f"  Chamfer Distance:   {results.get('overall', 'N/A'):.6f}")
        print('=' * 60)
        print(f"  Results saved to: {results_path}")
        print(f"  Culled mesh:       {output_dir / 'culled_mesh.ply'}")
        print(f"  Visualization:     {output_dir / f'vis_{int(args.scan_id):03d}_*.ply'}")
    else:
        print(f"[EVAL-DTU] Warning: 结果文件不存在: {results_path}")
        print("[EVAL-DTU] 评估可能失败，请检查错误信息")


if __name__ == '__main__':
    main()
