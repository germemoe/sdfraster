#!/bin/bash
# DTU 数据集 SDF 分裂 + Lambda 参数网格搜索脚本
# 测试 lambda_edge_sdf, lambda_normal_depth_med, lambda_mesh_depth, lambda_mesh_normal 的组合
# 同时启用 SDF 分裂使顶点数达到 90-100万

set -e

# 配置
OUTPUT_BASE="/home/jk/jk2/project/output/lambda_ablation_needsplit"
DATASET_BASE="/storage/jk2/dataset/DTU"
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# DTU场景列表
SCENES=(24 37 40 63 65 83 97 106)

# 场景对应的 SDF 分裂 ratio（根据当前顶点数计算）
# 目标: 95万顶点, SDF 分裂 20 次 (4000-14000, 每500迭代)
declare -A RATIOS
RATIOS[24]=0.024   # 58.4万 -> 95万, 1.63倍
RATIOS[37]=0.017   # 66.7万 -> 95万, 1.42倍
RATIOS[40]=0.005   # 86.1万 -> 95万, 1.10倍
RATIOS[63]=0.097    # 13.6万 -> 95万, 6.99倍
RATIOS[65]=0.03   # 38.9万 -> 95万, 2.44倍
RATIOS[83]=0.117    # 10.0万 -> 95万, 9.50倍
RATIOS[97]=0.038    # 44.7万 -> 95万, 2.13倍
RATIOS[106]=0.05  # 32.4万 -> 95万, 2.93倍

# Lambda参数搜索值
LAMBDA_EDGE_SDF_VALUES=(0 1e-4)
LAMBDA_NORMAL_DEPTH_MED_VALUES=(0 0.01)
LAMBDA_MESH_DEPTH_VALUES=(0 0.01)
LAMBDA_MESH_NORMAL_VALUES=(0 0.01)

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 生成所有任务
# 顺序：外层是 ablation 组合，内层是场景
# 优先级：1. lambda 都为 0.01  2. normal_depth 关闭的  3. 其他
declare -a TASKS
task_idx=0

# 添加任务的辅助函数
add_task() {
    local edge_sdf=$1
    local normal_depth_med=$2
    local mesh_depth=$3
    local mesh_normal=$4

    for scan in "${SCENES[@]}"; do
        exp_name="es${edge_sdf}_ndm${normal_depth_med}_md${mesh_depth}_mn${mesh_normal}"
        output_path="${OUTPUT_BASE}/scan${scan}/${exp_name}"

        # 跳过已完成的实验
        if [ -f "${output_path}/done.txt" ]; then
            echo "跳过已完成: scan${scan}/${exp_name}"
            continue
        fi

        TASKS[$task_idx]="${scan}|${edge_sdf}|${normal_depth_med}|${mesh_depth}|${mesh_normal}|${output_path}"
        task_idx=$((task_idx + 1))
    done
}

echo "生成任务列表..."

# 优先级1: lambda 都为 0.01
add_task 1e-4 0.01 0.01 0.01

# 优先级2: normal_depth 关闭的 (ndm=0)
for edge_sdf in "${LAMBDA_EDGE_SDF_VALUES[@]}"; do
    for mesh_depth in "${LAMBDA_MESH_DEPTH_VALUES[@]}"; do
        for mesh_normal in "${LAMBDA_MESH_NORMAL_VALUES[@]}"; do
            # 跳过已添加的组合
            if [ "$edge_sdf" = "1e-4" ] && [ "$mesh_depth" = "0.01" ] && [ "$mesh_normal" = "0.01" ]; then
                continue
            fi
            add_task "$edge_sdf" 0 "$mesh_depth" "$mesh_normal"
        done
    done
done

# 优先级3: 其他组合
for edge_sdf in "${LAMBDA_EDGE_SDF_VALUES[@]}"; do
    for normal_depth_med in "${LAMBDA_NORMAL_DEPTH_MED_VALUES[@]}"; do
        for mesh_depth in "${LAMBDA_MESH_DEPTH_VALUES[@]}"; do
            for mesh_normal in "${LAMBDA_MESH_NORMAL_VALUES[@]}"; do
                # 跳过已添加的组合 (1e-4/0.01/0.01/0.01 和 ndm=0 的)
                if [ "$edge_sdf" = "1e-4" ] && [ "$normal_depth_med" = "0.01" ] && [ "$mesh_depth" = "0.01" ] && [ "$mesh_normal" = "0.01" ]; then
                    continue
                fi
                if [ "$normal_depth_med" = "0" ]; then
                    continue
                fi
                add_task "$edge_sdf" "$normal_depth_med" "$mesh_depth" "$mesh_normal"
            done
        done
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "待运行任务数: $TOTAL_TASKS"
echo "使用GPU: ${GPUS[*]}"
echo ""

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "所有任务已完成!"
    exit 0
fi

# 运行单个任务的函数
run_single_task() {
    local gpu=$1
    local task_info=$2

    IFS='|' read -r scan edge_sdf normal_depth_med mesh_depth mesh_normal output_path <<< "$task_info"
    local exp_name="es${edge_sdf}_ndm${normal_depth_med}_md${mesh_depth}_mn${mesh_normal}"
    local ratio="${RATIOS[$scan]}"

    echo "[GPU $gpu] 开始: scan${scan}/${exp_name} (ratio=${ratio})"
    mkdir -p "$output_path"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$gpu \
    uv run train.py \
        --dataset_path "${DATASET_BASE}/scan${scan}" \
        --image_folder images \
        --output_path "$output_path" \
        --lambda_edge_sdf "$edge_sdf" \
        --lambda_normal_depth_med "$normal_depth_med" \
        --lambda_mesh_depth "$mesh_depth" \
        --lambda_mesh_normal "$mesh_normal" \
        --sdf_densify_enabled \
        --log2_hashmap_size 22 \
        --sdf_densify_ratio "$ratio" \
        --sdf_densify_start 4000 \
        --sdf_densify_end 14000 \
        > "${output_path}/train.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "done" > "${output_path}/done.txt"
        final_v=$(grep -oE "#V=[0-9]+" "${output_path}/train.log" | tail -1)
        echo "[GPU $gpu] ✓ 完成: scan${scan}/${exp_name} (${final_v})"
    else
        echo "[GPU $gpu] ✗ 失败: scan${scan}/${exp_name} (exit code: $exit_code)"
        echo "查看日志: ${output_path}/train.log"
    fi
    return $exit_code
}

# 简单的并行调度器
declare -A GPU_PIDS
declare -A GPU_TASKS
current_task=0
completed=0
failed=0

echo "开始运行实验..."
echo "=========================================="

while [ $completed -lt $TOTAL_TASKS ]; do
    # 检查每个GPU的状态
    for i in "${!GPUS[@]}"; do
        gpu=${GPUS[$i]}

        # 如果该GPU有正在运行的任务，检查是否完成
        if [ -n "${GPU_PIDS[$gpu]}" ]; then
            if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                # 任务完成
                wait "${GPU_PIDS[$gpu]}" 2>/dev/null || failed=$((failed + 1))
                completed=$((completed + 1))
                unset GPU_PIDS[$gpu]
                echo "进度: $completed/$TOTAL_TASKS (失败: $failed)"
            fi
        fi

        # 如果GPU空闲且还有任务，启动新任务
        if [ -z "${GPU_PIDS[$gpu]}" ] && [ $current_task -lt $TOTAL_TASKS ]; then
            run_single_task "$gpu" "${TASKS[$current_task]}" &
            GPU_PIDS[$gpu]=$!
            GPU_TASKS[$gpu]="${TASKS[$current_task]}"
            current_task=$((current_task + 1))
        fi
    done

    # 短暂等待避免CPU空转
    sleep 2
done

# 等待所有剩余任务
wait

echo "=========================================="
echo "全部完成! 成功: $((TOTAL_TASKS - failed)), 失败: $failed"
echo "结果保存在: $OUTPUT_BASE"
