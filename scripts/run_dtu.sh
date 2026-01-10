#!/bin/bash
# DTU 消融实验 - 严格参考 TNT 成功调度逻辑

# set -e  # 如果担心某个任务失败导致脚本整体退出，可以注释掉这一行
set +e 

OUTPUT_BASE="/data2/group_张举勇/yuhuimu/jk/output/DTU/norm_depth"
DATASET_BASE="/data2/group_张举勇/yuhuimu/jk/dataset/DTU"

# 自动解析可用 GPU (逻辑 ID)
IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
GPUS=()
for ((i=0; i<${#ADDR[@]}; i++)); do GPUS+=($i); done
NUM_GPUS=${#GPUS[@]}

# --- 场景配置字典 ---
declare -A EIKONAL CURVATURE S_VAR_LR SDF_RATIO NUM_SAMPLES
EIKONAL[37]=0.01; EIKONAL[63]=0.01; EIKONAL[65]=0.01; EIKONAL[69]=0.01; EIKONAL[83]=0.1; EIKONAL[110]=0.01
CURVATURE[37]=5e-06; CURVATURE[63]=5e-06; CURVATURE[65]=5e-06; CURVATURE[69]=5e-06; CURVATURE[83]=5e-06; CURVATURE[110]=0.0005
S_VAR_LR[37]=100; S_VAR_LR[63]=100; S_VAR_LR[65]=500; S_VAR_LR[69]=100; S_VAR_LR[83]=500; S_VAR_LR[110]=100
SDF_RATIO[37]=0.024; SDF_RATIO[63]=0.09; SDF_RATIO[65]=0.03; SDF_RATIO[69]=0.04; SDF_RATIO[83]=0.18; SDF_RATIO[110]=0.09
NUM_SAMPLES[37]=49; NUM_SAMPLES[63]=49; NUM_SAMPLES[65]=49; NUM_SAMPLES[69]=49; NUM_SAMPLES[83]=64; NUM_SAMPLES[110]=64

SCENES=(37 63 65 69 83 110)

mkdir -p "$OUTPUT_BASE"

# 运行单个任务函数
run_single_task() {
    local gpu=$1
    local task_str=$2
    
    # 解析任务字符串: "scan:nd:mn:md"
    IFS=':' read -r scan nd mn md <<< "$task_str"

    local exp_name="nd${nd}_mn${mn}_md${md}"
    local output_path="${OUTPUT_BASE}/${exp_name}/scan${scan}"
    local dataset_path="${DATASET_BASE}/scan${scan}"

    echo "[GPU $gpu] Starting Scan $scan ($exp_name)..."
    mkdir -p "$output_path"

    # 执行训练 - 明确指定变量
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run --env-file .env.4090 train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_normal_depth "$nd" \
        --lambda_mesh_normal "$mn" \
        --lambda_mesh_depth "$md" \
        --lambda_eikonal "${EIKONAL[$scan]}" \
        --lambda_curvature "${CURVATURE[$scan]}" \
        --s_var_lr_multiplier "${S_VAR_LR[$scan]}" \
        --sdf_densify_ratio "${SDF_RATIO[$scan]}" \
        --num_samples "${NUM_SAMPLES[$scan]}" \
        --eikonal_start 0 --eikonal_end 18000 \
        --use_geometric_init --sdf_densify_enabled --log2_hashmap_size 20 \
        --sdf_densify_start 4000 --sdf_densify_end 14000 \
        >"${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" >"${output_path}/done.txt"
        echo "[GPU $gpu] Scan $scan $exp_name Done!"
    else
        echo "[GPU $gpu] Scan $scan $exp_name Failed!"
    fi
    return $exit_code
}

# 收集任务
declare -a TASKS
for scan in "${SCENES[@]}"; do
    for nd in 0 0.01; do
        for mn in 0 0.01; do
            for md in 0 0.01; do
                task_str="$scan:$nd:$mn:$md"
                # 检查是否已完成
                exp_name="nd${nd}_mn${mn}_md${md}"
                if [ -f "${OUTPUT_BASE}/${exp_name}/scan${scan}/done.txt" ]; then
                    continue
                fi
                TASKS+=("$task_str")
            done
        done
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "========================================"
echo "Total tasks to run: $TOTAL_TASKS"
echo "GPUs: ${GPUS[*]}"
echo "========================================"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "All tasks already completed!"
    exit 0
fi

# 并行调度器 (完全参考你的 TNT 逻辑)
declare -A GPU_PIDS
current_task=0
completed=0
failed=0

while [ $completed -lt $TOTAL_TASKS ]; do
    for i in "${!GPUS[@]}"; do
        gpu=${GPUS[$i]}

        # 如果该GPU有任务，检查是否完成
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

wait
echo "All experiments finished!"