#!/bin/bash
#SBATCH -o cjk.%j.out
#SBATCH -J cjk
#SBATCH --qos=normal
#SBATCH -p A100         # 提交到 GPU 分区
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8       # 申请 1 块 GPU 卡

# 打印分配到的 GPU 信息
nvidia-smi

export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CPUS=$SLURM_CPUS_PER_TASK"

# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge
module load Anaconda3
source /opt/Software/Anaconda3/condabin/../etc/profile.d/conda.sh
source /home/chumeng/.bashrc
conda activate jk2
ml cuda/12.9.1

# 设置 CUDA 环境变量
export CUDA_HOME=/opt/Software/cuda/12.9.1
export PATH=/opt/Software/cuda/12.9.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/Software/cuda/12.9.1/lib64:$LD_LIBRARY_PATH

# 设置 CUDA 架构为 8.0 (A800/A100) - 确保编译时使用正确架构
export TORCH_CUDA_ARCH_LIST='8.0'
export UV_CACHE_DIR=/home/chumeng/.cache_80/uv

cd /data2/group_张举勇/chumeng/jk/radmesh

# 打印环境变量用于调试
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:0:150}..."

# srun --cpu-bind=cores --mem-bind=local python main_nocodec.py --start 46
# srun python main_nocodec.py --use_light_event_order --start 50
srun --export=ALL scripts/run_all_ablation.sh
echo "Job finished"
