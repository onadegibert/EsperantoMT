#!/bin/bash
#SBATCH --job-name=compute_metrics
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --output=logs/metrics_%J.out
#SBATCH --account=project_2005815
#SBATCH --partition=gputest
#SBATCH --ntasks=1

export HF_HOME=".hf_cache"
export TRANSFORMERS_CACHE=".hf_cache"
export HUGGINGFACE_HUB_CACHE=".hf_cache"

module load python-data
source ../venv/bin/activate

python scripts/compute_metrics.py 