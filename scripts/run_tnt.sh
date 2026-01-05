#!/bin/bash
# TNT (Tanks and Temples) 数据集训练脚本
# train.py 已经内置了 TNT 评估逻辑

set -e

# 配置
OUTPUT_BASE="/home/jk/jk2/project/output/tnt"
DATASET_BASE="/storage/jk2/dataset/TNT"
GPU=2

# TNT 场景列表
SCENES=(Caterpillar Courthouse Ignatius Meetingroom Truck)

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "TNT Dataset Training"
echo "========================================"
echo "Scenes: ${SCENES[*]}"
echo "GPU: $GPU"
echo ""

for scene in "${SCENES[@]}"; do
    output_path="${OUTPUT_BASE}/${scene}"
    dataset_path="${DATASET_BASE}/${scene}"

    if [ -f "${output_path}/done.txt" ]; then
        echo "[${scene}] Already done, skipping..."
        continue
    fi

    echo "[${scene}] Training..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$GPU \
    uv run train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --s_var_lr_multiplier 1 \
        --log2_hashmap_size 19 \
        --budget 1800000 \
        --sdf_eps 0.0 \
        --extract_tsdf \
        > "${output_path}/train.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        echo "[${scene}] Done!"
    else
        echo "[${scene}] Failed! See ${output_path}/train.log"
    fi
done

echo ""
echo "All scenes completed!"
