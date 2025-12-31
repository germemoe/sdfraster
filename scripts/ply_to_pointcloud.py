#!/usr/bin/env python3
"""
将训练保存的 ckpt.ply 转换为只包含顶点坐标的点云

用法:
    python scripts/ply_to_pointcloud.py --input output/test/ckpt.ply
    python scripts/ply_to_pointcloud.py --input output/test/ckpt.ply --output pointcloud.ply
    python scripts/ply_to_pointcloud.py --input output/test/ckpt.ply --exclude_external
"""
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def convert_to_pointcloud(input_path: str, output_path: str = None, exclude_external: bool = False):
    """
    将 ckpt.ply 转换为只包含顶点的点云

    Args:
        input_path: 输入 ply 文件路径
        output_path: 输出 ply 文件路径，默认为 input_path 同目录下的 pointcloud.ply
        exclude_external: 是否排除外部顶点（ext_vertices，通常是最后 1000 个点）
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / "pointcloud.ply"
    else:
        output_path = Path(output_path)

    print(f"Reading: {input_path}")
    plydata = PlyData.read(str(input_path))

    # 获取顶点数据
    vertex_data = plydata['vertex'].data

    # 提取 x, y, z 坐标
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']

    num_vertices = len(x)
    print(f"Total vertices: {num_vertices}")

    # 排除外部顶点（如果需要）
    if exclude_external:
        # 外部顶点通常是最后 1000 个（fibonacci_spiral_on_sphere 生成的辅助点）
        num_external = 1000
        if num_vertices > num_external:
            num_internal = num_vertices - num_external
            x = x[:num_internal]
            y = y[:num_internal]
            z = z[:num_internal]
            print(f"Excluding {num_external} external vertices, keeping {num_internal} internal vertices")
        else:
            print(f"Warning: Total vertices ({num_vertices}) <= external count ({num_external}), keeping all")

    # 创建新的顶点数组（只有 x, y, z）
    vertices = np.zeros(len(x), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertices['x'] = x
    vertices['y'] = y
    vertices['z'] = z

    # 创建 PLY 元素
    vertex_element = PlyElement.describe(vertices, 'vertex')

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([vertex_element], text=False).write(str(output_path))

    print(f"Saved pointcloud to: {output_path}")
    print(f"  Vertices: {len(vertices)}")


def main():
    parser = argparse.ArgumentParser(description='将 ckpt.ply 转换为只包含顶点的点云')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入 ply 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出 ply 文件路径（默认为同目录下的 pointcloud.ply）')
    parser.add_argument('--exclude_external', '-e', action='store_true',
                        help='排除外部辅助顶点（最后 1000 个点）')
    args = parser.parse_args()

    convert_to_pointcloud(args.input, args.output, args.exclude_external)


if __name__ == '__main__':
    main()
