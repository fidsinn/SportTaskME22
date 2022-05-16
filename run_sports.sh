#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=sport
#SBATCH --partition=clara-job
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00

module load PyTorch/1.9.0-fosscuda-2020b
#module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
module load scikit-learn
#pip install matplotlib
#module load matplotlib
module load OpenCV

srun python main.py