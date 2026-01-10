#!/bin/bash
# TNT (Tanks and Temples) 数据集训练脚本
# train.py 已经内置了 TNT 评估逻辑

set -e

# 配置
OUTPUT_BASE="/data2/group_张举勇/yuhuimu/jk/output/TNT"
DATASET_BASE="/data2/group_张举勇/yuhuimu/jk/dataset/TNT"
GPUS=(0 1)
NUM_GPUS=${#GPUS[@]}

# TNT 场景列表
SCENES=(Barn Caterpillar Courthouse Ignatius Meetingroom Truck)

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "TNT Dataset Training (Parallel)"
echo "========================================"
echo "Scenes: ${SCENES[*]}"
echo "GPUs: ${GPUS[*]}"
echo ""

# 运行单个任务的函数
run_single_task() {
  local gpu=$1
  local scene=$2
  local output_path="${OUTPUT_BASE}/${scene}"
  local dataset_path="${DATASET_BASE}/${scene}"

  echo "[GPU $gpu] Starting ${scene}..."
  mkdir -p "$output_path"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run --env-file .env.4090 train.py \
    --dataset_path "$dataset_path" \
    --image_folder images \
    --output_path "$output_path" \
    --s_var_lr_multiplier 1 \
    --log2_hashmap_size 19 \
    --budget 3000000 \
    --sdf_eps 0.0 \
    --extract_tsdf \
    >"${output_path}/train.log" 2>&1

  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "done" >"${output_path}/done.txt"
    echo "[GPU $gpu] ${scene} Done!"
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
  TASKS[$task_idx]="$scene"
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
      run_single_task "$gpu" "${TASKS[$current_task]}" &
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
echo "All scenes completed! Success: $((TOTAL_TASKS - failed)), Failed: $failed"
echo "========================================"
