#!/bin/bash

# Loss 权重消融实验脚本
# 参数：lambda_normal_depth_med, lambda_edge_sdf, lambda_mesh_normal, lambda_mesh_depth
# 值：0.05, 0.01
# 可用 GPU: 0, 2, 3, 5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_CMD="uv run train.py --eval \
    --dataset_path /home/jk/data/dataset/mipnerf360/bicycle \
    --image_folder images_4 \
    --log2_hashmap_size 19"

OUTPUT_BASE="output/sdf/mipnerf/bicycle_hash19"

# ============== GPU 0: lambda_normal_depth_med ==============
run_gpu0() {
    export CUDA_VISIBLE_DEVICES=0

    # echo "[GPU0] Running: lambda_normal_depth_med=0.05"
    # $BASE_CMD \
    #     --output_path ${OUTPUT_BASE}_normal_depth_med_0.05 \
    #     --lambda_normal_depth_med 0.05

    echo "[GPU0] Running: lambda_normal_depth_med=0.01"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}_normal_depth_med_0.01 \
        --lambda_normal_depth_med 0.01
}

# ============== GPU 2: lambda_edge_sdf ==============
run_gpu2() {
    export CUDA_VISIBLE_DEVICES=5

    echo "[GPU2] Running: lambda_edge_sdf=0.0001"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}_edge_sdf_0.0001 \
        --lambda_edge_sdf 0.0001

    echo "[GPU2] Running: lambda_edge_sdf=0.00001"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}_edge_sdf_0.00001 \
        --lambda_edge_sdf 0.00001
}

# ============== GPU 3: lambda_mesh_normal ==============
run_gpu3() {
    export CUDA_VISIBLE_DEVICES=7

    echo "[GPU3] Running: lambda_mesh_normal=0.05"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}_mesh_normal_0.05 \
        --lambda_mesh_normal 0.05

    echo "[GPU3] Running: lambda_mesh_normal=0.01"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}_mesh_normal_0.01 \
        --lambda_mesh_normal 0.01
}

# ============== GPU 5: lambda_mesh_depth ==============
run_gpu5() {
    export CUDA_VISIBLE_DEVICES=2

    echo "[GPU5] Running: lambda_mesh_depth=0.05"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}_mesh_depth_0.05 \
        --lambda_mesh_depth 0.05

    echo "[GPU5] Running: lambda_mesh_depth=0.01"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}_mesh_depth_0.01 \
        --lambda_mesh_depth 0.01
}

# ============== 主入口 ==============
echo "Starting loss ablation experiments..."
echo "GPU 0: lambda_normal_depth_med (0.05, 0.01)"
echo "GPU 2: lambda_edge_sdf (0.05, 0.01)"
echo "GPU 3: lambda_mesh_normal (0.05, 0.01)"
echo "GPU 5: lambda_mesh_depth (0.05, 0.01)"
echo ""

# # 并行启动四个 GPU 的实验
# run_gpu0 &
# PID_GPU0=$!

# run_gpu2 &
# PID_GPU2=$!

# run_gpu3 &
# PID_GPU3=$!

# run_gpu5 &
# PID_GPU5=$!

# # 等待所有进程完成
# wait $PID_GPU0
# echo "GPU 0 (lambda_normal_depth_med) experiments finished."

# wait $PID_GPU2
# echo "GPU 2 (lambda_edge_sdf) experiments finished."

# wait $PID_GPU3
# echo "GPU 3 (lambda_mesh_normal) experiments finished."

# wait $PID_GPU5
# echo "GPU 5 (lambda_mesh_depth) experiments finished."

# echo "All loss ablation experiments completed!"
# echo "Results saved in: ${OUTPUT_BASE}_*"
