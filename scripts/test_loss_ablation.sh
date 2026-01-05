#!/bin/bash
# Loss ablation test for scan105
# edge_sdf = 0 (fixed)
# normal_depth, mesh_normal, mesh_depth: 0.01 or 0
# Exclude all-zero case (7 combinations)

set -e

OUTPUT_BASE="/home/jk/jk2/project/output/loss_ablation"
DATASET_BASE="/storage/jk2/dataset/DTU"
GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}

SCAN=105

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Loss Ablation Test (scan${SCAN})"
echo "========================================"
echo "GPUs: ${GPUS[*]}"
echo ""

# 运行单个任务的函数
run_single_task() {
    local gpu=$1
    local nd=$2
    local mn=$3
    local md=$4
    local output_path="${OUTPUT_BASE}/scan${SCAN}/nd${nd}_mn${mn}_md${md}"
    local dataset_path="${DATASET_BASE}/scan${SCAN}"

    echo "[GPU $gpu] Starting nd=${nd}, mn=${mn}, md=${md}..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_edge_sdf 0 \
        --lambda_normal_depth_med "$nd" \
        --lambda_mesh_normal "$mn" \
        --lambda_mesh_depth "$md" \
        --log2_hashmap_size 22 \
        > "${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        echo "[GPU $gpu] nd${nd}_mn${mn}_md${md} Done!"
    else
        echo "[GPU $gpu] nd${nd}_mn${mn}_md${md} Failed! See ${output_path}/train.log"
    fi
    return $exit_code
}

# 定义测试组合: nd mn md (排除全0)
declare -a CASES=(
    "0.01 0 0"
    "0 0.01 0"
    "0 0 0.01"
    "0.01 0.01 0"
    "0.01 0 0.01"
    "0 0.01 0.01"
    "0.01 0.01 0.01"
)

# 收集待运行的任务
declare -a TASKS
task_idx=0
for case in "${CASES[@]}"; do
    read -r nd mn md <<< "$case"
    output_path="${OUTPUT_BASE}/scan${SCAN}/nd${nd}_mn${mn}_md${md}"
    if [ -f "${output_path}/done.txt" ]; then
        echo "[nd${nd}_mn${mn}_md${md}] Already done, skipping..."
        continue
    fi
    TASKS[$task_idx]="$nd $mn $md"
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
    # 检查每个GPU的状态
    for i in "${!GPUS[@]}"; do
        gpu=${GPUS[$i]}

        # 如果该GPU有正在运行的任务，检查是否完成
        if [ -n "${GPU_PIDS[$gpu]}" ]; then
            if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                wait "${GPU_PIDS[$gpu]}" 2>/dev/null || failed=$((failed + 1))
                completed=$((completed + 1))
                unset GPU_PIDS[$gpu]
                echo "Progress: $completed/$TOTAL_TASKS (failed: $failed)"
            fi
        fi

        # 如果GPU空闲且还有任务，启动新任务
        if [ -z "${GPU_PIDS[$gpu]}" ] && [ $current_task -lt $TOTAL_TASKS ]; then
            read -r nd mn md <<< "${TASKS[$current_task]}"
            run_single_task "$gpu" "$nd" "$mn" "$md" &
            GPU_PIDS[$gpu]=$!
            current_task=$((current_task + 1))
        fi
    done

    sleep 2
done

# 等待所有剩余任务
wait

echo ""
echo "========================================"
echo "All tests completed! Success: $((TOTAL_TASKS - failed)), Failed: $failed"
echo "========================================"

# 汇总结果
echo ""
echo "Results:"
for case in "${CASES[@]}"; do
    read -r nd mn md <<< "$case"
    log="${OUTPUT_BASE}/scan${SCAN}/nd${nd}_mn${mn}_md${md}/train.log"
    if [ -f "$log" ]; then
        psnr=$(grep -oE "Mean: [0-9.]+" "$log" | tail -1 | awk '{print $2}')
        echo "  nd${nd}_mn${mn}_md${md}: PSNR=${psnr}"
    fi
done
