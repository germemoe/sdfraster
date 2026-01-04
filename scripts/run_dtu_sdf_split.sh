#!/bin/bash
# DTU 数据集重训脚本 - 启用 SDF 分裂使顶点数达到 90-100万
# 基于 es1e-4_ndm0.01_md0.01_mn0.01 配置

set -e

OUTPUT_BASE="/home/jk/jk2/project/output/lambda_ablation_sdf"
DATASET_BASE="/storage/jk2/dataset/DTU"
GPU=2

# 场景及其 SDF 分裂 ratio（根据当前顶点数计算）
# 目标: 95万顶点, SDF 分裂 20 次 (4000-14000, 每500迭代)
# ratio = (目标/当前)^0.05 - 1
declare -A RATIOS
RATIOS[24]=0.025   # 58.4万 -> 95万, 1.63倍
RATIOS[37]=0.018   # 66.7万 -> 95万, 1.42倍
RATIOS[40]=0.005   # 86.1万 -> 95万, 1.10倍
RATIOS[63]=0.10    # 13.6万 -> 95万, 6.99倍
RATIOS[69]=0.045   # 38.9万 -> 95万, 2.44倍
RATIOS[83]=0.12    # 10.0万 -> 95万, 9.50倍
RATIOS[97]=0.04    # 44.7万 -> 95万, 2.13倍
RATIOS[105]=0.08   # 20.9万 -> 95万, 4.55倍
RATIOS[106]=0.055  # 32.4万 -> 95万, 2.93倍
RATIOS[110]=0.12   # 10.1万 -> 95万, 9.41倍

SCENES=(24 37 40 63 69 83 97 105 106 110)

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "DTU SDF Split Training"
echo "========================================"
echo "Scenes: ${SCENES[*]}"
echo "GPU: $GPU"
echo ""

for scan in "${SCENES[@]}"; do
    ratio=${RATIOS[$scan]}
    output_path="${OUTPUT_BASE}/scan${scan}"
    dataset_path="${DATASET_BASE}/scan${scan}"

    if [ -f "${output_path}/done.txt" ]; then
        echo "[scan${scan}] Already done, skipping..."
        continue
    fi

    echo "[scan${scan}] Training with sdf_densify_ratio=${ratio}..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$GPU \
    uv run train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_edge_sdf 1e-4 \
        --lambda_normal_depth_med 0.01 \
        --lambda_mesh_depth 0.01 \
        --lambda_mesh_normal 0.01 \
        --sdf_densify_enabled \
        --sdf_densify_ratio "$ratio" \
        --sdf_densify_start 4000 \
        --sdf_densify_end 14000 \
        > "${output_path}/train.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        # 打印最终顶点数
        final_v=$(grep -oE "#V=[0-9]+" "${output_path}/train.log" | tail -1)
        echo "[scan${scan}] Done! Final ${final_v}"
    else
        echo "[scan${scan}] Failed! See ${output_path}/train.log"
    fi
done

echo ""
echo "========================================"
echo "All scenes completed!"
echo "========================================"

# 汇总结果
echo ""
echo "Final vertex counts:"
for scan in "${SCENES[@]}"; do
    log="${OUTPUT_BASE}/scan${scan}/train.log"
    if [ -f "$log" ]; then
        v=$(grep -oE "#V=[0-9]+" "$log" | tail -1)
        echo "  scan${scan}: ${v}"
    fi
done
