#!/bin/bash
# MipNeRF360 数据集训练脚本
# GPU 2,3,6,7 并行，不跑 bicycle

set -e

OUTPUT_BASE="/data2/group_张举勇/yuhuimu/jk/output/360_normdepth"
DATASET_BASE="/data2/group_张举勇/yuhuimu/jk/dataset/360"
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# 场景及其 image_folder 设置（参考 run_m360_2.csv）
declare -A IMAGE_FOLDERS
IMAGE_FOLDERS[bicycle]=images_4
IMAGE_FOLDERS[garden]=images_4
IMAGE_FOLDERS[stump]=images_4
IMAGE_FOLDERS[treehill]=images_4
IMAGE_FOLDERS[flowers]=images_4
IMAGE_FOLDERS[room]=images_2
IMAGE_FOLDERS[counter]=images_2
IMAGE_FOLDERS[kitchen]=images_2
IMAGE_FOLDERS[bonsai]=images_2

# 不跑 bicycle
SCENES=(bicycle garden stump treehill flowers room counter kitchen bonsai)

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "MipNeRF360 Dataset Training (Parallel)"
echo "========================================"
echo "Scenes: ${SCENES[*]}"
echo "GPUs: ${GPUS[*]}"
echo ""

# 运行单个任务的函数
run_single_task() {
    local gpu=$1
    local scene=$2
    local image_folder=$3
    local output_path="${OUTPUT_BASE}/${scene}"
    local dataset_path="${DATASET_BASE}/${scene}"

    echo "[GPU $gpu] Starting ${scene} (image_folder=${image_folder})..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run --env-file .env.4090 train.py \
        --dataset_path "$dataset_path" \
        --image_folder "$image_folder" \
        --output_path "$output_path" \
        --log2_hashmap_size 19 \
        --sdf_eps 0.0 \
        --extract_tsdf \
        --lambda_normal_depth_med 0.01 \
        --lambda_mesh_depth 0.01 \
        --lambda_mesh_normal 0.01 \
        --s_var_lr_multiplier 1 \
        --budget 3000000 \
        --eval \
        > "${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        final_v=$(grep -oE "#V=[0-9]+" "${output_path}/train.log" | tail -1)
        echo "[GPU $gpu] ${scene} Done! ${final_v}"
    else
        echo "[GPU $gpu] ${scene} Failed! See ${output_path}/train.log"
    fi
    return $exit_code
}

# 收集待运行的任务
declare -a TASKS
task_idx=0
for scene in "${SCENES[@]}"; do
    output_path="${OUTPUT_BASE}/${scene}"
    if [ -f "${output_path}/done.txt" ]; then
        echo "[${scene}] Already done, skipping..."
        continue
    fi
    TASKS[$task_idx]="${scene}|${IMAGE_FOLDERS[$scene]}"
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
            IFS='|' read -r scene image_folder <<< "${TASKS[$current_task]}"
            run_single_task "$gpu" "$scene" "$image_folder" &
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
