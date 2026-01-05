#!/bin/bash
# Gradient TV Loss 测试脚本
# 测试场景：scan40, 69, 110, 97, 83
# GPU 4, 7 并行

set -e

OUTPUT_BASE="/home/jk/jk2/project/output/lambda_ablation_gradtv"
DATASET_BASE="/storage/jk2/dataset/DTU"
GPUS=(4 7)
NUM_GPUS=${#GPUS[@]}

declare -A RATIOS


RATIOS[40]=0.005   # 86.1万 -> 95万, 1.10倍
RATIOS[69]=0.032   # 38.9万 -> 95万, 2.44倍
RATIOS[83]=0.117    # 10.0万 -> 95万, 9.50倍
RATIOS[97]=0.038    # 44.7万 -> 95万, 2.13倍
RATIOS[110]=0.117  # 32.4万 -> 95万, 2.93倍

SCENES=(40 69 110 97 83)

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Gradient TV Test (Parallel)"
echo "========================================"
echo "Scenes: ${SCENES[*]}"
echo "GPUs: ${GPUS[*]}"
echo ""

# 运行单个任务的函数
run_single_task() {
    local gpu=$1
    local scan=$2
    local ratio=$3
    local output_path="${OUTPUT_BASE}/scan${scan}_gradtv"
    local dataset_path="${DATASET_BASE}/scan${scan}"

    echo "[GPU $gpu] Starting scan${scan}..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_edge_sdf 0.0001 \
        --lambda_normal_depth_med 0.01 \
        --lambda_mesh_depth 0.01 \
        --lambda_mesh_normal 0.01 \
        --sdf_densify_enabled \
        --log2_hashmap_size 21 \
        --sdf_densify_ratio "$ratio" \
        --sdf_densify_start 4000 \
        --sdf_densify_end 14000 \
        --lambda_gradient_tv 0.0001 \
        --gradient_tv_start 4000 \
        > "${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        final_v=$(grep -oE "#V=[0-9]+" "${output_path}/train.log" | tail -1)
        echo "[GPU $gpu] scan${scan} Done! ${final_v}"
    else
        echo "[GPU $gpu] scan${scan} Failed! See ${output_path}/train.log"
    fi
    return $exit_code
}

# 收集待运行的任务
declare -a TASKS
task_idx=0
for scan in "${SCENES[@]}"; do
    output_path="${OUTPUT_BASE}/scan${scan}_gradtv"
    if [ -f "${output_path}/done.txt" ]; then
        echo "[scan${scan}] Already done, skipping..."
        continue
    fi
    TASKS[$task_idx]="$scan|${RATIOS[$scan]}"
    task_idx=$((task_idx + 1))
done

TOTAL_TASKS=${#TASKS[@]}
echo "Tasks to run: $TOTAL_TASKS"
echo ""

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "All tasks already completed!"
    exit 0
fi

# 并行调度器
declare -A GPU_PIDS
current_task=0
completed=0
failed=0

while [ $completed -lt $TOTAL_TASKS ]; do
    for i in "${!GPUS[@]}"; do
        gpu=${GPUS[$i]}

        if [ -n "${GPU_PIDS[$gpu]}" ]; then
            if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                wait "${GPU_PIDS[$gpu]}" 2>/dev/null || failed=$((failed + 1))
                completed=$((completed + 1))
                unset GPU_PIDS[$gpu]
                echo "Progress: $completed/$TOTAL_TASKS (failed: $failed)"
            fi
        fi

        if [ -z "${GPU_PIDS[$gpu]}" ] && [ $current_task -lt $TOTAL_TASKS ]; then
            IFS='|' read -r scan ratio <<< "${TASKS[$current_task]}"
            run_single_task "$gpu" "$scan" "$ratio" &
            GPU_PIDS[$gpu]=$!
            current_task=$((current_task + 1))
        fi
    done

    sleep 2
done

wait

echo ""
echo "========================================"
echo "All completed! Success: $((TOTAL_TASKS - failed)), Failed: $failed"
echo "========================================"
