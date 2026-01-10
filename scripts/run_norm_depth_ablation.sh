#!/bin/bash
# Normal/Depth Loss 消融实验
# 基于每个场景的最佳 eikonal + curvature 配置
# 测试 lambda_normal_depth, lambda_mesh_normal, lambda_mesh_depth 的 0/0.01 组合
# 每个场景 2^3 = 8 个实验

set -e

OUTPUT_BASE="/home/jk/jk2/project/output/norm_depth_ablation"
DATASET_BASE="/storage/jk2/dataset/DTU"
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# 场景列表
SCENES=(37 63 65 69 83 110)

# 每个场景的最佳 eikonal 配置
declare -A EIKONAL
EIKONAL[37]=0.01
EIKONAL[63]=0.01
EIKONAL[65]=0.01
EIKONAL[69]=0.01
EIKONAL[83]=0.1
EIKONAL[110]=0.01

# 每个场景的最佳 curvature 配置
declare -A CURVATURE
CURVATURE[37]=5e-06
CURVATURE[63]=5e-06
CURVATURE[65]=5e-06
CURVATURE[69]=5e-06
CURVATURE[83]=5e-06
CURVATURE[110]=0.0005

# 每个场景的 s_var_lr_multiplier
declare -A S_VAR_LR
S_VAR_LR[37]=100
S_VAR_LR[63]=100
S_VAR_LR[65]=500
S_VAR_LR[69]=100
S_VAR_LR[83]=500
S_VAR_LR[110]=100

# 每个场景的 sdf_densify_ratio
declare -A SDF_RATIO
SDF_RATIO[37]=0.024
SDF_RATIO[63]=0.09
SDF_RATIO[65]=0.03
SDF_RATIO[69]=0.04
SDF_RATIO[83]=0.18
SDF_RATIO[110]=0.09

# 每个场景的 num_samples
declare -A NUM_SAMPLES
NUM_SAMPLES[37]=49
NUM_SAMPLES[63]=49
NUM_SAMPLES[65]=49
NUM_SAMPLES[69]=49
NUM_SAMPLES[83]=64
NUM_SAMPLES[110]=64

mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Normal/Depth Loss Ablation"
echo "========================================"
echo "GPUs: ${GPUS[*]}"
echo "Scenes: ${SCENES[*]}"
echo ""

# 运行单个任务的函数
run_single_task() {
    local gpu=$1
    local scan=$2
    local nd=$3      # lambda_normal_depth
    local mn=$4      # lambda_mesh_normal
    local md=$5      # lambda_mesh_depth

    local eik=${EIKONAL[$scan]}
    local curv=${CURVATURE[$scan]}
    local s_var=${S_VAR_LR[$scan]}
    local ratio=${SDF_RATIO[$scan]}
    local num_samp=${NUM_SAMPLES[$scan]}

    local exp_name="nd${nd}_mn${mn}_md${md}"
    local output_path="${OUTPUT_BASE}/${exp_name}/scan${scan}"
    local dataset_path="${DATASET_BASE}/scan${scan}"

    # 跳过已完成的实验
    if [ -f "${output_path}/done.txt" ]; then
        echo "[GPU $gpu] ${exp_name}/scan${scan} already done, skipping..."
        return 0
    fi

    echo "[GPU $gpu] Starting ${exp_name}/scan${scan} (eik=$eik, curv=$curv)..."
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run train.py \
        --dataset_path "$dataset_path" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_eikonal "$eik" \
        --lambda_curvature "$curv" \
        --eikonal_start 0 \
        --eikonal_end 18000 \
        --s_var_lr_multiplier "$s_var" \
        --sdf_densify_ratio "$ratio" \
        --num_samples "$num_samp" \
        --lambda_normal_depth "$nd" \
        --lambda_mesh_normal "$mn" \
        --lambda_mesh_depth "$md" \
        --lambda_edge_sdf 0 \
        --lambda_normal_depth_med 0.0 \
        --use_geometric_init \
        --sdf_densify_enabled \
        --log2_hashmap_size 20 \
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
declare -a TASKS=()

# lambda_normal_depth, lambda_mesh_normal, lambda_mesh_depth 的组合
ND_VALUES=(0 0.01)
MN_VALUES=(0 0.01)
MD_VALUES=(0 0.01)

for scan in "${SCENES[@]}"; do
    for nd in "${ND_VALUES[@]}"; do
        for mn in "${MN_VALUES[@]}"; do
            for md in "${MD_VALUES[@]}"; do
                TASKS+=("${scan}:${nd}:${mn}:${md}")
            done
        done
    done
done

echo "Total tasks: ${#TASKS[@]}"
echo ""

# 并行执行任务
task_idx=0
while [ $task_idx -lt ${#TASKS[@]} ]; do
    # 启动一批任务
    pids=()
    for ((i=0; i<NUM_GPUS && task_idx<${#TASKS[@]}; i++, task_idx++)); do
        task=${TASKS[$task_idx]}
        IFS=':' read -r scan nd mn md <<< "$task"
        gpu=${GPUS[$i]}

        run_single_task "$gpu" "$scan" "$nd" "$mn" "$md" &
        pids+=($!)
    done

    # 等待这批任务完成
    for pid in "${pids[@]}"; do
        wait $pid
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
