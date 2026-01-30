#!/bin/bash
#SBATCH --job-name=nllb_eval
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G            # Increased for 3.3B model
#SBATCH --output=logs/nllb_eval_%A_%a.out
#SBATCH --account=project_2005815
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --array=0-23

export HF_HOME=".hf_cache"
export TRANSFORMERS_CACHE=".hf_cache"
export HUGGINGFACE_HUB_CACHE=".hf_cache"

module load python-data
source venv/bin/activate

# 6 Language pairs
LANGPAIRS=("epo-eng" "epo-spa" "epo-cat" "eng-epo" "spa-epo" "cat-epo")

# 4 Models
MODELS=(
  "facebook/nllb-200-distilled-1.3B"
  "facebook/nllb-200-distilled-600M"
  "facebook/nllb-200-3.3B"
  "facebook/nllb-200-1.3B"
)

# Logic: SLURM_ARRAY_TASK_ID 0-3 = Langpair 0, 4-7 = Langpair 1, etc.
LP_INDEX=$(( SLURM_ARRAY_TASK_ID / 4 ))
MD_INDEX=$(( SLURM_ARRAY_TASK_ID % 4 ))

LANGPAIR=${LANGPAIRS[$LP_INDEX]}
MODEL=${MODELS[$MD_INDEX]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Evaluating $MODEL on $LANGPAIR"

python scripts/eval_nllb.py "$LANGPAIR" "$MODEL"