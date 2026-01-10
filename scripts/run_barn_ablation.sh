#!/bin/bash
# Barn 场景消融实验脚本 - 测试单个 loss 的影响
# 适配 SLURM 环境：自动检测分配的 GPU 数量

set -e

# ============ 配置区域 ============
OUTPUT_BASE="/data2/group_张举勇/yuhuimu/jk/output/TNT_ablation"
DATASET_PATH="/data2/group_张举勇/yuhuimu/jk/dataset/TNT/Barn"
ENV_FILE=".env.a100"  # 根据机器修改: .env.4090 或 .env.a100

# 自动检测可用 GPU 数量（兼容 SLURM 环境）
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # SLURM 环境：从 CUDA_VISIBLE_DEVICES 解析 GPU 数量
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "Detected SLURM environment: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    # 非 SLURM 环境：使用 nvidia-smi 检测
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi
# 生成 GPU 索引数组 (0, 1, 2, ..., N-1)
GPUS=($(seq 0 $((NUM_GPUS - 1))))
echo "Using ${NUM_GPUS} GPUs: ${GPUS[*]}"

# ============ 任务定义 ============
# 格式: "配置名|额外参数"
declare -a TASKS

# edge_sdf loss: 1e-4, 1e-5, 1e-6
TASKS+=("edge_1e-4|--lambda_edge_sdf 1e-4")
TASKS+=("edge_1e-5|--lambda_edge_sdf 1e-5")
TASKS+=("edge_1e-6|--lambda_edge_sdf 1e-6")

# dist loss: 1e-4, 1e-2
TASKS+=("dist_1e-4|--lambda_dist 1e-4")
TASKS+=("dist_1e-2|--lambda_dist 1e-2")

# lambda_normal_depth_med: 0.01, 0.05
TASKS+=("normal_depth_med_0.01|--lambda_normal_depth_med 0.01")
TASKS+=("normal_depth_med_0.05|--lambda_normal_depth_med 0.05")

# lambda_mesh_depth: 0.01, 0.05
TASKS+=("mesh_depth_0.01|--lambda_mesh_depth 0.01")
TASKS+=("mesh_depth_0.05|--lambda_mesh_depth 0.05")

# lambda_mesh_normal: 0.01, 0.05
TASKS+=("mesh_normal_0.01|--lambda_mesh_normal 0.01")
TASKS+=("mesh_normal_0.05|--lambda_mesh_normal 0.05")

# ============ 运行函数 ============
run_task() {
    local gpu=$1
    local task_str=$2

    IFS='|' read -r config extra_args <<< "$task_str"

    local output_path="${OUTPUT_BASE}/${config}"

    echo "[GPU $gpu] Starting Barn/${config}..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run --env-file "$ENV_FILE" train.py \
        --dataset_path "$DATASET_PATH" \
        --image_folder images \
        --output_path "$output_path" \
        --log2_hashmap_size 19 \
        --sdf_eps 0.0 \
        --extract_tsdf \
        --s_var_lr_multiplier 1 \
        --budget 3000000 \
        $extra_args \
        > "${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        final_v=$(grep -oE "#V=[0-9]+" "${output_path}/train.log" | tail -1)
        echo "[GPU $gpu] Barn/${config} Done! ${final_v}"
    else
        echo "[GPU $gpu] Barn/${config} Failed! See ${output_path}/train.log"
    fi
    return $exit_code
}

# ============ 过滤已完成任务 ============
declare -a PENDING_TASKS
for task in "${TASKS[@]}"; do
    IFS='|' read -r config extra_args <<< "$task"
    output_path="${OUTPUT_BASE}/${config}"
    if [ -f "${output_path}/done.txt" ]; then
        echo "[SKIP] Barn/${config} already done"
    else
        PENDING_TASKS+=("$task")
    fi
done

TOTAL_TASKS=${#PENDING_TASKS[@]}

echo ""
echo "========================================"
echo "Barn Ablation Study"
echo "========================================"
echo "Total tasks: ${#TASKS[@]}, Pending: $TOTAL_TASKS"
echo "GPUs: ${GPUS[*]}"
echo ""

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "All tasks already completed!"
    exit 0
fi

# ============ 并行调度器 ============
declare -A GPU_PIDS
current_task=0
completed=0
failed=0

while [ $completed -lt $TOTAL_TASKS ]; do
    for gpu in "${GPUS[@]}"; do
        # 检查GPU上的任务是否完成
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
            run_task "$gpu" "${PENDING_TASKS[$current_task]}" &
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
