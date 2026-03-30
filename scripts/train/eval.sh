#!/bin/bash
#SBATCH --job-name="eval_eo_multi"
#SBATCH --account=project_2005815
#SBATCH --output=logs/eval_eo_multi_%j.out
#SBATCH --error=logs/eval_eo_multi_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1000

set -euo pipefail
set -x

module load gcc cuda

model_dir="models/esenca_eo_shuf"
model_path="${model_dir}/model.npz.best-chrf.npz"
res_file="${model_dir}/results.txt"
vocab="models/esenca_eo/vocab.spm"

# 3 translation directions
LANG_DIRS=(
  "spa-epo"
  "cat-epo"
  "eng-epo"
)

# Activate sacrebleu env once
source ../MTM25/venv/bin/activate

for LANGDIR in "${LANG_DIRS[@]}"; do
    # split "spa-arg" into src="spa", tgt="arg"
    src=${LANGDIR%-*}
    tgt=${LANGDIR#*-}

    src_file="data/test.${src}"
    trg_file="data/test.${tgt}"
    hyp_file="${model_dir}/test.${LANGDIR}.${tgt}.out"

    if [[ ! -f "$hyp_file" ]]; then

        echo "Translating $LANGDIR with ${model_path}..."
        mkdir -p "$(dirname "$hyp_file")"
        : > "$hyp_file"  # truncate

        # | sed "s/^/>>${src}<< >>${tgt}<< /"  \
        cat "$src_file" | /scratch/project_2005815/members/degibert/MTM25/marian/build/marian-decoder \
                --output "$hyp_file" --normalize --mini-batch 64 --maxi-batch 100 --maxi-batch-sort src \
                -m "$model_path" \
                --vocabs "$vocab" "$vocab" \
                --log "${model_dir}/test.${LANGDIR}.${tgt}.log" \
                --devices 0

        echo "Decoding done for $LANGDIR."
    fi

    echo "Scoring with sacrebleu..."
    bleu=$(sacrebleu "$trg_file" -i "$hyp_file" -m bleu -w 2 --score-only)
    chrf=$(sacrebleu "$trg_file" -i "$hyp_file" -m chrf -w 2 --score-only)

    mkdir -p "$(dirname "$res_file")"
    if [[ ! -f "$res_file" ]]; then
        printf "Language Pair\tBLEU\tChrF\n" > "$res_file"
    fi

    printf "%s\t%s\t%s\n" "$LANGDIR" "$bleu" "$chrf" >> "$res_file"
done

model_dir="models/eo_esenca_shuf"
model_path="${model_dir}/model.npz.best-chrf.npz"
res_file="${model_dir}/results.txt"

# 3 translation directions
LANG_DIRS=(
  "epo-spa"
  "epo-cat"
  "epo-eng"
)

for LANGDIR in "${LANG_DIRS[@]}"; do
    # split "spa-arg" into src="spa", tgt="arg"
    src=${LANGDIR%-*}
    tgt=${LANGDIR#*-}

    src_file="data/test.${src}"
    trg_file="data/test.${tgt}"
    hyp_file="${model_dir}/test.${LANGDIR}.${tgt}.out"

    if [[ ! -f "$hyp_file" ]]; then

        echo "Translating $LANGDIR with ${model_path}..."
        mkdir -p "$(dirname "$hyp_file")"
        : > "$hyp_file"  # truncate

        cat "$src_file" | sed "s/^/>>${tgt}<< /"  \
        | /scratch/project_2005815/members/degibert/MTM25/marian/build/marian-decoder \
                --output "$hyp_file" --normalize --mini-batch 64 --maxi-batch 100 --maxi-batch-sort src \
                -m "$model_path" \
                --vocabs "$vocab" "$vocab" \
                --log "${model_dir}/test.${LANGDIR}.${tgt}.log" \
                --devices 0

        echo "Decoding done for $LANGDIR."
    fi

    echo "Scoring with sacrebleu..."
    bleu=$(sacrebleu "$trg_file" -i "$hyp_file" -m bleu -w 2 --score-only)
    chrf=$(sacrebleu "$trg_file" -i "$hyp_file" -m chrf -w 2 --score-only)

    mkdir -p "$(dirname "$res_file")"
    if [[ ! -f "$res_file" ]]; then
        printf "Language Pair\tBLEU\tChrF\n" > "$res_file"
    fi

    printf "%s\t%s\t%s\n" "$LANGDIR" "$bleu" "$chrf" >> "$res_file"
done
