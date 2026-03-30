#!/bin/bash
#SBATCH --job-name="convert_%j.sh"
#SBATCH --account=project_2011987
#SBATCH --output=logs/convert_%j.log
#SBATCH --error=logs/convert_%j.log
#SBATCH --time=00:15:00
#SBATCH --partition=small
#SBATCH --mem=32G
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1


langpair="esenca_eo_shuf"

source ../MTM25/venv/bin/activate
mkdir  models/hf/$langpair
python scripts/convert_marian_to_hf.py --src models/$langpair/ --dest models/hf/$langpair