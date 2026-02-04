#!/bin/bash
MX_IN="data/metricx_inputs"
MX_OUT="data/metricx_scores"
mkdir -p $MX_OUT

for file in "$MX_IN"/*.jsonl; do
    name=$(basename "$file")
    out="$MX_OUT/$name"

    if [ -f "$out" ]; then
        echo "Skipping $name — output already exists."
        continue
    fi

    echo "Evaluating $name with MetricX-24..."
    python -m metricx24.predict \
        --tokenizer google/mt5-xl \
        --model_name_or_path google/metricx-24-hybrid-large-v2p6 \
        --input_file "$file" \
        --output_file "$out" \
        --batch_size 64 \
        --max_input_length 1024
done