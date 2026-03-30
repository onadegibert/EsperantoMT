#!/bin/bash
#SBATCH --job-name=opusfilter
#SBATCH --output=logs/opusfilter_%j.out
#SBATCH --error=logs/opusfilter_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --account=project_2005815

source venv/bin/activate
opusfilter --n-jobs 1 scripts/opusfilter_config_es-eo.yml