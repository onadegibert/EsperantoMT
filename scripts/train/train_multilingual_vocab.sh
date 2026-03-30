#!/bin/bash
#SBATCH --job-name="vocab"
#SBATCH --account=project_2005815
#SBATCH --output=logs/vocab_%j.out
#SBATCH --error=logs/vocab_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G

set -x
set -euo pipefail

module load gcc cuda

mkdir -p logs models/multi


# SentencePiece model prefix (no extension here)
vocab_prefix="models/multi/vocab"
vocab="${vocab_prefix}.spm"

# -------- Train SentencePiece vocab --------
shuf -n 400000 data/lowres/train.spa-xx.spa > tmp.spa
shuf -n 400000 data/lowres/train.spa-ara.ara > tmp.ara
shuf -n 400000 data/lowres/train.spa-arg.arg > tmp.arg
shuf -n 400000 data/lowres/train.spa-ast.ast > tmp.ast
shuf -n 400000 data/lowres/train.spa-oci.oci > tmp.oci
shuf -n 400000 data/es-eo/clean/train.eo > tmp.epo
shuf -n 400000 data/ca-eo/clean/train.ca > tmp.cat

cat tmp.spa tmp.ara tmp.arg tmp.ast tmp.oci tmp.epo tmp.cat > data/spm_data.multi.txt
rm tmp*

/scratch/project_2005815/members/degibert/MTM25/marian/build/spm_train \
  --user_defined_symbols=">>ara<<",">>arg<<",">>ast<<",">>oci<<",">>cat<<",">>spa<<",">>epo<<" \
  --bos_id=-1 --eos_id=0 --unk_id=1 \
  --input=data/spm_data.multi.txt \
  --vocab_size=32000 \
  --character_coverage=0.9995 \
  --model_prefix="$vocab_prefix" --byte_fallback

mv "${vocab_prefix}.model" $vocab

# # -------- Run Marian training --------

# /scratch/project_2005815/members/degibert/MTM25/marian/build/marian -c scripts/marian_config.yml \
#   --model "models/es-xx/model.npz" \
#   --train-sets "$train_src_all" "$train_trg_all" \
#   --valid-sets "$dev_src_all" "$dev_trg_all" \
#   --vocabs "$vocab" "$vocab" \
#   --log "models/es-xx/train.log" \
#   --valid-log "models/es-xx/valid.log" \
#   --devices 0 1 2 3