#!/bin/bash

# 消融实验脚本：测试 prune、分裂、剔除四面体的不同参数
# GPU 4, 5, 6, 7 四卡并行运行
#
# 布尔参数说明（Args类的特殊处理）：
# - 默认 False 的参数：传 --flag 表示 True
# 所以：
# - --sdf_densify_enabled (默认False): 传这个参数表示启用
# - --prune_enabled (默认False): 传这个参数表示启用

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_CMD="uv run train.py \
    --dataset_path /storage/jk/dataset/DTU/scan37 \
    --image_folder images \
    --lambda_edge_sdf 1e-3 \
    --lambda_normal_depth_med 0.01 \
    --lambda_mesh_depth 0.01"

OUTPUT_BASE="newoutput/sdf/DTU/ablation"

# ============== GPU 4: Prune 实验 ==============
run_gpu4() {
    export CUDA_VISIBLE_DEVICES=4

    # 1. Prune 实验：不同 contrib_threshold
    echo "[GPU4] Running: prune_contrib_0.005"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/prune_contrib_0.005 \
        --prune_enabled \
        --prune_contrib_threshold 0.005 \
        --prune_sdf_far_threshold 0.1

    echo "[GPU4] Running: prune_contrib_0.02"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/prune_contrib_0.02 \
        --prune_enabled \
        --prune_contrib_threshold 0.02 \
        --prune_sdf_far_threshold 0.1

    # 2. Prune 实验：不同 sdf_far_threshold
    echo "[GPU4] Running: prune_sdf_far_0.05"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/prune_sdf_far_0.05 \
        --prune_enabled \
        --prune_contrib_threshold 0.01 \
        --prune_sdf_far_threshold 0.05

    echo "[GPU4] Running: prune_sdf_far_0.2"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/prune_sdf_far_0.2 \
        --prune_enabled \
        --prune_contrib_threshold 0.01 \
        --prune_sdf_far_threshold 0.2
}

# ============== GPU 5: SDF Densify 实验（禁用 prune）==============
run_gpu5() {
    export CUDA_VISIBLE_DEVICES=5

    # SDF Densify 实验：不同 ratio
    echo "[GPU5] Running: sdf_densify_ratio_0.2"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/sdf_densify_ratio_0.2 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.2 \
        --sdf_densify_threshold 0.02

    echo "[GPU5] Running: sdf_densify_ratio_0.33"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/sdf_densify_ratio_0.33 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.33 \
        --sdf_densify_threshold 0.02

    echo "[GPU5] Running: sdf_densify_ratio_0.5"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/sdf_densify_ratio_0.5 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.5 \
        --sdf_densify_threshold 0.02

    # SDF Densify 实验：不同 threshold
    echo "[GPU5] Running: sdf_densify_thresh_0.01"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/sdf_densify_thresh_0.01 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.33 \
        --sdf_densify_threshold 0.01
}

# ============== GPU 6: 组合实验（Prune + SDF Densify，错开250代）==============
run_gpu6() {
    export CUDA_VISIBLE_DEVICES=6

    # 组合实验：Prune + SDF Densify，prune_start 错开 250 代
    echo "[GPU6] Running: combined_offset250"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/combined_offset250 \
        --prune_enabled \
        --prune_contrib_threshold 0.01 \
        --prune_sdf_far_threshold 0.1 \
        --prune_start 4250 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.33 \
        --sdf_densify_threshold 0.02

    # 组合实验：不同 prune 参数
    echo "[GPU6] Running: combined_prune_strict"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/combined_prune_strict \
        --prune_enabled \
        --prune_contrib_threshold 0.005 \
        --prune_sdf_far_threshold 0.05 \
        --prune_start 4250 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.33 \
        --sdf_densify_threshold 0.02

    # 组合实验：不同 sdf_densify 参数
    echo "[GPU6] Running: combined_sdf_dense"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/combined_sdf_dense \
        --prune_enabled \
        --prune_contrib_threshold 0.01 \
        --prune_sdf_far_threshold 0.1 \
        --prune_start 4250 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.5 \
        --sdf_densify_threshold 0.01
}

# ============== GPU 7: Cull 实验 + 其他 ==============
run_gpu7() {
    export CUDA_VISIBLE_DEVICES=7

    # 四面体剔除实验：不同 contrib_threshold
    echo "[GPU7] Running: cull_contrib_0.005"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/cull_contrib_0.005 \
        --contrib_threshold 0.005

    echo "[GPU7] Running: cull_contrib_0.01"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/cull_contrib_0.01 \
        --contrib_threshold 0.01

    echo "[GPU7] Running: cull_contrib_0.02"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/cull_contrib_0.02 \
        --contrib_threshold 0.02

    # SDF Densify 实验：不同 threshold
    echo "[GPU7] Running: sdf_densify_thresh_0.05"
    $BASE_CMD \
        --output_path ${OUTPUT_BASE}/sdf_densify_thresh_0.05 \
        --sdf_densify_enabled \
        --sdf_densify_ratio 0.33 \
        --sdf_densify_threshold 0.05
}

# ============== 主入口 ==============
echo "Starting ablation experiments on GPU 4, 5, 6, 7..."
echo "GPU 4: Prune experiments (contrib_threshold, sdf_far_threshold)"
echo "GPU 5: SDF Densify experiments (ratio, threshold)"
echo "GPU 6: Combined experiments (Prune + SDF Densify, offset 250 iters)"
echo "GPU 7: Cull experiments + extra sdf_densify"
echo ""

# 并行启动四个 GPU 的实验
run_gpu4 &
PID_GPU4=$!

run_gpu5 &
PID_GPU5=$!

run_gpu6 &
PID_GPU6=$!

run_gpu7 &
PID_GPU7=$!

# 等待所有进程完成
wait $PID_GPU4
echo "GPU 4 experiments finished."

wait $PID_GPU5
echo "GPU 5 experiments finished."

wait $PID_GPU6
echo "GPU 6 experiments finished."

wait $PID_GPU7
echo "GPU 7 experiments finished."

echo "All ablation experiments completed!"
