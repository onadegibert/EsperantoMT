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

#TODO remove * from Apertium
# Ensure MetricX is in your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/../metricx

#python scripts/compute_metrics_1.py

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

bash scripts/compute_metricx.sh

python scripts/compute_metrics_2.py
