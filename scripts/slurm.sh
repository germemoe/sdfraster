#!/bin/bash
#SBATCH -o blenderproc.%j.out
#SBATCH -J blenderproc
#SBATCH --qos=normal
#SBATCH -p L40         # 提交到 GPU 分区
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8        # 申请 1 块 GPU 卡

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
cd /data2/group_张举勇/yuhuimu/jk/radmesh
export UV_CACHE_DIR=/home/yuhuimu/.cache/uv
# srun --cpu-bind=cores --mem-bind=local python main_nocodec.py --start 46
# srun python main_nocodec.py --use_light_event_order --start 50
# 并行运行三个脚本，分配不同的GPU子集
# slurm分配的GPU通过CUDA_VISIBLE_DEVICES传递，脚本内部使用相对索引
IFS=',' read -ra SLURM_GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "SLURM allocated GPUs: ${SLURM_GPUS[*]}"

# run_m360.sh 使用前3张卡 (索引0,1,2)
CUDA_VISIBLE_DEVICES="${SLURM_GPUS[0]},${SLURM_GPUS[1]}" scripts/run_m360.sh &

# run_m3601.sh 使用中间3张卡 (索引3,4,5)
CUDA_VISIBLE_DEVICES="${SLURM_GPUS[2]},${SLURM_GPUS[3]},${SLURM_GPUS[4]},${SLURM_GPUS[5]}" scripts/run_m3601.sh &

# run_tnt.sh 使用最后2张卡 (索引6,7)
CUDA_VISIBLE_DEVICES="${SLURM_GPUS[6]},${SLURM_GPUS[7]}" scripts/run_tnt.sh &

wait
echo "Job finished"
