#!/bin/bash
# 测试各个 loss 的脚本
# 每个实验单独开启一个 loss，其他 loss 保持关闭
# 每个实验使用不同的 GPU 并行运行

# 基础配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET_PATH="/home/jk/data/dataset/DTU/scan37"
IMAGE_FOLDER="images"
OUTPUT_BASE="output/sdf/DTU/newnew"

# 基础命令
BASE_CMD="uv run train.py --dataset_path ${DATASET_PATH} --image_folder ${IMAGE_FOLDER}"

echo "=========================================="
echo "开始 Loss 测试实验（并行）"
echo "=========================================="

# # 1. Eikonal Loss (GPU 0)
# echo "[1/8] Eikonal Loss (GPU 0)"
# CUDA_VISIBLE_DEVICES=0 $BASE_CMD --output_path ${OUTPUT_BASE}/eikonal \
#     --lambda_eikonal 0.1 &

# # 2. Curvature Loss (GPU 1)
# echo "[2/8] Curvature Loss (GPU 1)"
# CUDA_VISIBLE_DEVICES=1 $BASE_CMD --output_path ${OUTPUT_BASE}/curvature \
#     --lambda_curvature 5e-6 &

# 3. Mesh Normal Loss (GPU 2)
echo "[3/8] Mesh Normal Loss (GPU 2)"
CUDA_VISIBLE_DEVICES=1 $BASE_CMD --output_path ${OUTPUT_BASE}/mesh_normaldexp \
    --lambda_mesh_normal 0.05 \
    --mesh_normal_start 5000 &

# 4. Mesh Depth Loss (GPU 3)
echo "[4/8] Mesh Depth Loss (GPU 3)"
CUDA_VISIBLE_DEVICES=3 $BASE_CMD --output_path ${OUTPUT_BASE}/mesh_depth \
    --lambda_mesh_depth 0.1 \
    --mesh_normal_start 5000 

# 5. Normal-Depth Consistency Loss (GPU 4)
echo "[5/8] Normal-Depth Consistency Loss (GPU 4)"
CUDA_VISIBLE_DEVICES=1 $BASE_CMD --output_path ${OUTPUT_BASE}/normal_depth \
    --lambda_normal_depth 0.05 &

# 6. Normal-Depth Median Consistency Loss (GPU 5)
echo "[6/8] Normal-Depth Median Consistency Loss (GPU 5)"
CUDA_VISIBLE_DEVICES=3 $BASE_CMD --output_path ${OUTPUT_BASE}/normal_depth_med \
    --lambda_normal_depth_med 0.05 

# 7. Mask Loss (GPU 6)
echo "[7/8] Mask Loss (GPU 6)"
CUDA_VISIBLE_DEVICES=1 $BASE_CMD --output_path ${OUTPUT_BASE}/mask \
    --lambda_mask 0.1 \
    --mask_from 0 \
    --mask_until 20000 &

# 8. T Concentration Loss (GPU 7)
echo "[8/8] T Concentration Loss (GPU 7)"
CUDA_VISIBLE_DEVICES=3 $BASE_CMD --output_path ${OUTPUT_BASE}/T_concen \
    --lambda_T_concen 0.01 &

echo ""
echo "所有实验已启动，等待完成..."
wait

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "结果保存在: ${OUTPUT_BASE}/"
echo "  - eikonal/        : Eikonal loss (GPU 0)"
echo "  - curvature/      : Curvature loss (GPU 1)"
echo "  - mesh_normal/    : Mesh normal loss (GPU 2)"
echo "  - mesh_depth/     : Mesh depth loss (GPU 3)"
echo "  - normal_depth/   : Normal-depth consistency (GPU 4)"
echo "  - normal_depth_med/: Normal-depth consistency med (GPU 5)"
echo "  - mask/           : Mask loss (GPU 6)"
echo "  - T_concen/       : T concentration loss (GPU 7)"
