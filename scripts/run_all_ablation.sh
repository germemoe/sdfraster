#!/bin/bash
# 统一消融实验脚本 - 多 GPU 并行调度
# 包含: 360/TNT 的 基础/dist/normdepth/dist+normdepth 配置
# 适配 SLURM 环境：自动检测分配的 GPU 数量

set -e

# ============ 配置区域 ============
OUTPUT_BASE="/data2/group_张举勇/yuhuimu/jk/output"
DATASET_360="/data2/group_张举勇/yuhuimu/jk/dataset/360"
DATASET_TNT="/data2/group_张举勇/yuhuimu/jk/dataset/TNT"
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

# ============ 场景定义 ============
# 360 场景及其 image_folder
declare -A M360_IMAGE_FOLDERS
M360_IMAGE_FOLDERS[bicycle]=images_4
M360_IMAGE_FOLDERS[garden]=images_4
M360_IMAGE_FOLDERS[stump]=images_4
M360_IMAGE_FOLDERS[treehill]=images_4
M360_IMAGE_FOLDERS[flowers]=images_4
M360_IMAGE_FOLDERS[room]=images_2
M360_IMAGE_FOLDERS[counter]=images_2
M360_IMAGE_FOLDERS[kitchen]=images_2
M360_IMAGE_FOLDERS[bonsai]=images_2

M360_ALL=(bicycle garden stump treehill flowers room counter kitchen bonsai)
TNT_ALL=(Barn Caterpillar Courthouse Ignatius Meetingroom Truck)

# ============ 任务定义 ============
# 格式: "数据集类型|场景|配置名|输出子目录|额外参数"
declare -a TASKS

add_360_task() {
    local scene=$1
    local config=$2
    local output_dir=$3
    local extra_args=$4
    TASKS+=("360|${scene}|${config}|${output_dir}|${extra_args}")
}

add_tnt_task() {
    local scene=$1
    local config=$2
    local output_dir=$3
    local extra_args=$4
    TASKS+=("TNT|${scene}|${config}|${output_dir}|${extra_args}")
}

# --- 360 基础: bicycle ---
add_360_task "bicycle" "base" "360" ""

# --- 360 dist: bonsai, kitchen ---
add_360_task "bonsai" "dist" "360_dist" "--lambda_dist 1e-3"
add_360_task "kitchen" "dist" "360_dist" "--lambda_dist 1e-3"

# --- 360 normdepth: bicycle, garden, treehill ---
add_360_task "bicycle" "normdepth" "360_normdepth" "--lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"
add_360_task "garden" "normdepth" "360_normdepth" "--lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"
add_360_task "treehill" "normdepth" "360_normdepth" "--lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"

# --- TNT 基础: Barn, Caterpillar ---
add_tnt_task "Barn" "base" "TNT" ""
add_tnt_task "Caterpillar" "base" "TNT" ""

# --- TNT dist: Barn, Caterpillar ---
add_tnt_task "Barn" "dist" "TNT_dist" "--lambda_dist 1e-3"
add_tnt_task "Caterpillar" "dist" "TNT_dist" "--lambda_dist 1e-3"

# --- TNT normdepth: Barn, Caterpillar, Ignatius ---
add_tnt_task "Barn" "normdepth" "TNT_normdepth" "--lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"
add_tnt_task "Caterpillar" "normdepth" "TNT_normdepth" "--lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"
add_tnt_task "Ignatius" "normdepth" "TNT_normdepth" "--lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"

# --- 360 dist+normdepth: 全部 ---
for scene in "${M360_ALL[@]}"; do
    add_360_task "$scene" "dist_normdepth" "360_dist_normdepth" "--lambda_dist 1e-3 --lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"
done

# --- TNT dist+normdepth: 全部 ---
for scene in "${TNT_ALL[@]}"; do
    add_tnt_task "$scene" "dist_normdepth" "TNT_dist_normdepth" "--lambda_dist 1e-3 --lambda_normal_depth_med 0.01 --lambda_mesh_depth 0.01 --lambda_mesh_normal 0.01"
done

# ============ 运行函数 ============
run_task() {
    local gpu=$1
    local task_str=$2

    IFS='|' read -r dataset_type scene config output_dir extra_args <<< "$task_str"

    if [ "$dataset_type" == "360" ]; then
        local dataset_path="${DATASET_360}/${scene}"
        local image_folder="${M360_IMAGE_FOLDERS[$scene]}"
        local eval_flag="--eval"
    else
        local dataset_path="${DATASET_TNT}/${scene}"
        local image_folder="images"
        local eval_flag=""
    fi

    local output_path="${OUTPUT_BASE}/${output_dir}/${scene}"

    echo "[GPU $gpu] Starting ${output_dir}/${scene} (${config})..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run --env-file "$ENV_FILE" train.py \
        --dataset_path "$dataset_path" \
        --image_folder "$image_folder" \
        --output_path "$output_path" \
        --log2_hashmap_size 19 \
        --sdf_eps 0.0 \
        --extract_tsdf \
        --s_var_lr_multiplier 1 \
        --budget 3000000 \
        $eval_flag \
        $extra_args \
        > "${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        final_v=$(grep -oE "#V=[0-9]+" "${output_path}/train.log" | tail -1)
        echo "[GPU $gpu] ${output_dir}/${scene} Done! ${final_v}"
    else
        echo "[GPU $gpu] ${output_dir}/${scene} Failed! See ${output_path}/train.log"
    fi
    return $exit_code
}

# ============ 过滤已完成任务 ============
declare -a PENDING_TASKS
for task in "${TASKS[@]}"; do
    IFS='|' read -r dataset_type scene config output_dir extra_args <<< "$task"
    output_path="${OUTPUT_BASE}/${output_dir}/${scene}"
    if [ -f "${output_path}/done.txt" ]; then
        echo "[SKIP] ${output_dir}/${scene} already done"
    else
        PENDING_TASKS+=("$task")
    fi
done

TOTAL_TASKS=${#PENDING_TASKS[@]}

echo ""
echo "========================================"
echo "Ablation Study - Unified Runner"
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
            sleep 3  # 避免多个任务同时初始化 CUDA 导致竞争
        fi
    done

    sleep 2
done

wait

echo ""
echo "========================================"
echo "All completed! Success: $((TOTAL_TASKS - failed)), Failed: $failed"
echo "========================================"
