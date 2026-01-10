#!/bin/bash
#SBATCH -o blenderproc.%j.out
#SBATCH -J blenderproc
#SBATCH --qos=normal
#SBATCH -p RTX4090          # 提交到 GPU 分区
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:7        # 申请 1 块 GPU 卡

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
source /home/yuhuimu/.bashrc
conda activate jk
export UV_CACHE_DIR=/home/yuhuimu/.cache/uv
cd /data2/group_张举勇/yuhuimu/jk/radmesh
# srun --cpu-bind=cores --mem-bind=local python main_nocodec.py --start 46
# srun python main_nocodec.py --use_light_event_order --start 50
bash scripts/run_dtu.sh
echo "Job finished"
