#!/bin/bash
# Eikonal Loss + Curvature Loss 消融实验
# 测试组合：
#   eikonal: 0.1, 0.01, 1.0
#   curvature: 5e-5, 5e-6
# 其中 eikonal=0.1 + curvature=5e-6 只跑 69, 83 场景

set -e

OUTPUT_BASE="/home/jk/jk2/project/output/eikonal_curv_ablation"
DATASET_BASE="/storage/jk2/dataset/DTU"
GPUS=(3 6)
NUM_GPUS=${#GPUS[@]}

# 全部场景
ALL_SCENES=(37 40 63 65 69 83 97 110)

# 场景对应的 SDF 分裂 ratio
declare -A RATIOS
RATIOS[24]=0.036
RATIOS[37]=0.024
RATIOS[40]=0.006
RATIOS[63]=0.09
RATIOS[65]=0.03
RATIOS[69]=0.04
RATIOS[83]=0.18
RATIOS[97]=0.06
RATIOS[105]=0.04
RATIOS[106]=0.025
RATIOS[110]=0.09

# 场景对应的 s_var 学习率倍率
declare -A S_VAR_LRS
S_VAR_LRS[37]=100
S_VAR_LRS[40]=100
S_VAR_LRS[63]=100
S_VAR_LRS[65]=500
S_VAR_LRS[69]=100
S_VAR_LRS[83]=500
S_VAR_LRS[97]=500
S_VAR_LRS[110]=100

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Eikonal + Curvature Ablation (Parallel)"
echo "========================================"
echo "GPUs: ${GPUS[*]}"
echo ""

# 运行单个任务的函数
run_single_task() {
    local gpu=$1
    local scan=$2
    local ratio=$3
    local s_var_lr=$4
    local eikonal=$5
    local curvature=$6
    local exp_name="eik${eikonal}_curv${curvature}"
    local output_path="${OUTPUT_BASE}/${exp_name}/scan${scan}"
    local dataset_path="${DATASET_BASE}/scan${scan}"

    echo "[GPU $gpu] Starting ${exp_name}/scan${scan}..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_edge_sdf 0 \
        --lambda_normal_depth_med 0.0 \
        --lambda_mesh_depth 0.0 \
        --lambda_mesh_normal 0.0 \
        --lambda_eikonal "$eikonal" \
        --lambda_curvature "$curvature" \
        --eikonal_start 0 \
        --eikonal_end 18000 \
        --s_var_lr_multiplier "$s_var_lr" \
        --use_geometric_init \
        --sdf_densify_enabled \
        --log2_hashmap_size 20 \
        --sdf_densify_ratio "$ratio" \
        --sdf_densify_start 4000 \
        --sdf_densify_end 14000 \
        > "${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        final_v=$(grep -oE "#V=[0-9]+" "${output_path}/train.log" | tail -1)
        echo "[GPU $gpu] ${exp_name}/scan${scan} Done! ${final_v}"
    else
        echo "[GPU $gpu] ${exp_name}/scan${scan} Failed! See ${output_path}/train.log"
    fi
    return $exit_code
}

# 收集所有任务
declare -a TASKS
task_idx=0

# 定义实验组合
# 格式: eikonal|curvature|scenes (scenes 用逗号分隔)
EXPERIMENTS=(
    "0.1|5e-5|all"
    "0.1|5e-6|69,83"
    "0.01|5e-5|all"
    "0.01|5e-6|all"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r eikonal curvature scene_list <<< "$exp"

    if [ "$scene_list" = "all" ]; then
        scenes=("${ALL_SCENES[@]}")
    else
        IFS=',' read -ra scenes <<< "$scene_list"
    fi

    exp_name="eik${eikonal}_curv${curvature}"
    echo "Experiment: $exp_name, scenes: ${scenes[*]}"

    for scan in "${scenes[@]}"; do
        output_path="${OUTPUT_BASE}/${exp_name}/scan${scan}"
        if [ -f "${output_path}/done.txt" ]; then
            echo "  [${exp_name}/scan${scan}] Already done, skipping..."
            continue
        fi
        TASKS[$task_idx]="${scan}|${RATIOS[$scan]}|${S_VAR_LRS[$scan]}|${eikonal}|${curvature}"
        task_idx=$((task_idx + 1))
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "Total tasks to run: $TOTAL_TASKS"
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
            IFS='|' read -r scan ratio s_var_lr eikonal curvature <<< "${TASKS[$current_task]}"
            run_single_task "$gpu" "$scan" "$ratio" "$s_var_lr" "$eikonal" "$curvature" &
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
