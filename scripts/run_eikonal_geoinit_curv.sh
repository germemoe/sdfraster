#!/bin/bash
# Eikonal Loss + 几何初始化测试脚本
# 测试场景：24,37,40,63,65,83,97,105,106,110
# 开启：eikonal loss, normal_depth_med, mesh_depth, mesh_normal, 几何初始化
# 关闭：edge_sdf

set -e

OUTPUT_BASE="/home/jk/jk2/project/output/eikonal_geoinit_curv"
DATASET_BASE="/storage/jk2/dataset/DTU"
GPUS=(2 7)
NUM_GPUS=${#GPUS[@]}

SCENES=(24 37 40 63 65 83 97 105 106 110)

# 场景对应的 SDF 分裂 ratio
declare -A RATIOS
RATIOS[24]=0.012
RATIOS[37]=0.008
RATIOS[40]=0.002
RATIOS[63]=0.048
RATIOS[65]=0.015
RATIOS[83]=0.058
RATIOS[97]=0.019
RATIOS[105]=0.04
RATIOS[106]=0.025
RATIOS[110]=0.062

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Eikonal + Geometric Init Test (Parallel)"
echo "========================================"
echo "Scenes: ${SCENES[*]}"
echo "GPUs: ${GPUS[*]}"
echo ""

# 运行单个任务的函数
run_single_task() {
    local gpu=$1
    local scan=$2
    local ratio=$3
    local output_path="${OUTPUT_BASE}/scan${scan}"
    local dataset_path="${DATASET_BASE}/scan${scan}"

    echo "[GPU $gpu] Starting scan${scan} (ratio=${ratio})..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_edge_sdf 0 \
        --lambda_normal_depth_med 0.01 \
        --lambda_mesh_depth 0.01 \
        --lambda_mesh_normal 0.01 \
        --lambda_eikonal 0.1 \
        --lambda_curvature 5e-4 \
        --eikonal_start 0 \
        --eikonal_end 18000 \
        --s_var_lr_multiplier 100 \
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
    output_path="${OUTPUT_BASE}/scan${scan}"
    if [ -f "${output_path}/done.txt" ]; then
        echo "[scan${scan}] Already done, skipping..."
        continue
    fi
    TASKS[$task_idx]="${scan}|${RATIOS[$scan]}"
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
