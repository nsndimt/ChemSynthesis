#!/bin/bash
#SBATCH --job-name="llamafactory"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for pytorch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=4
#SBATCH --constraint="projects&scratch"
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account="Your Slurm Account"    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --time 02:00:00
#SBATCH --mail-user=zhangyue@udel.edu
#SBATCH --output="%x-%j.out"

echo "job is starting on `hostname`"
echo "jobid: $SLURM_JOB_ID"
echo "landing dir: $(pwd)"

module reset
module load anaconda3_gpu
module load cuda/12.4.0
module list

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
echo "conda is at $(which conda)"
conda env list

echo "gcc version: $(gcc --version) location: $(which gcc)"
echo "nvcc version: $(nvcc --version) location: $(which nvcc)"
echo "cuda home: $CUDA_HOME"
echo "path: $PATH"
echo "ld_library: $LD_LIBRARY_PATH"

export OMP_NUM_THREADS=16

conda deactivate
conda deactivate
conda activate llamafactory
echo "python is at $(which python)"

source stage1.sh