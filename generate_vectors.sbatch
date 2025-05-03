#!/bin/bash
#SBATCH --job-name=wiki_vectorize
#SBATCH --output=wiki_vectorize.out
#SBATCH --error=wiki_vectorize.err
#SBATCH -p nvidia-a100
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Load modules or activate environment
eval "$(conda shell.bash hook)"
conda activate pyda

echo "Loaded python version $(python --version) at $(which python)"

nvidia-smi

echo "Starting vectorization on GPU at $(date)"

python -u generate_vectors.py

echo "Finished at $(date)"

