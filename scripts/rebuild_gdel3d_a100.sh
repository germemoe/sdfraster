#!/bin/bash
# 重新编译 gDel3D for sm_80 架构 (A100/A800)
# 需要在 GPU 节点上运行此脚本

set -e

echo "=== 检查 GPU ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv

echo ""
echo "=== 设置环境变量 ==="
export TORCH_CUDA_ARCH_LIST='8.0'
export CUDA_HOME=/opt/Software/cuda/12.9.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CUDA_HOME=$CUDA_HOME"

echo ""
echo "=== 激活虚拟环境 ==="
source /data2/group_张举勇/yuhuimu/jk/radmesh/.venv_a100/bin/activate

echo ""
echo "=== 进入 gDel3D 源码目录 ==="
cd /home/yuhuimu/.cache_80/uv/git-v0/checkouts/805bf3e1e6f3a0d6/245c717

# 清理旧的编译文件
echo ""
echo "=== 清理旧的编译文件 ==="
rm -rf build/ dist/ *.egg-info/ src/*.egg-info/ 2>/dev/null || true

echo ""
echo "=== 开始编译 gDel3D (sm_80) ==="
python -m pip install . --no-build-isolation --force-reinstall -v

echo ""
echo "=== 验证编译结果 ==="
SO_FILE=$(find /data2/group_张举勇/yuhuimu/jk/radmesh/.venv_a100/lib/python3.10/site-packages/gdel3d -name "*.so" | head -1)
if [ -n "$SO_FILE" ]; then
    echo "检查 $SO_FILE 的 CUDA 架构:"
    cuobjdump --list-elf "$SO_FILE"
else
    echo "警告: 未找到 .so 文件"
fi

echo ""
echo "=== 测试 gDel3D ==="
python -c "import gdel3d; print('gDel3D 导入成功!')"

echo ""
echo "=== 完成 ==="
