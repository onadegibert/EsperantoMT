import os
import json
import pandas as pd
import torch
import evaluate
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
import sys
sys.path.append(os.path.expanduser("../metricx"))
from metricx24 import predict as metricx_predict

# --- Setup Models & Metrics ---
# SacreBLEU and chrF++
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
all_results = []

# COMET (Unbabel/wmt22-comet-da is the standard reference-based model)
comet_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_path)

# MetricX-24 XXL
metricx_model = "google/metricx-24-hybrid-large-v2p6-bfloat16"

# Configuration
PRED_DIR = "data/predictions/"
lang_codes = {"eng": "eng_Latn", "epo": "epo_Latn", "cat": "cat_Latn", "spa": "spa_Latn"}

for pred_file in os.listdir(PRED_DIR):
    
    langpair = pred_file.split(".")[0]
    system = pred_file.split(".")[1]    # "marian"
    src_iso, tgt_iso = langpair.split("-")
    src_lang, tgt_lang = lang_codes[src_iso], lang_codes[tgt_iso]
    
    # Load FLORES+ and predictions
    src_ds = load_dataset("openlanguagedata/flores_plus", src_lang, split="devtest")
    tgt_ds = load_dataset("openlanguagedata/flores_plus", tgt_lang, split="devtest")
    with open(os.path.join(PRED_DIR, pred_file), "r", encoding="utf-8") as f:
        hypotheses = [line.strip() for line in f]

    sources = [x["text"] for x in src_ds]
    references = [x["text"] for x in tgt_ds]

    # --- 1. Compute SacreBLEU & chrF++ ---
    bleu_score = bleu.compute(predictions=hypotheses, references=[[r] for r in references])['score']
    chrf_score = chrf.compute(predictions=hypotheses, references=[[r] for r in references], word_order=2)['score']

    # --- 2. Compute COMET ---
    # COMET requires a list of dicts with 'src', 'mt', 'ref'
    comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
    comet_score = comet_output.system_score * 100 # Often scaled to 0-100

    # --- 3. Compute MetricX-24 XXL ---
    metricx_data = [{"source": s, "hypothesis": h, "reference": r} for s, h, r in zip(sources, hypotheses, references)]
    metricx_scores = metricx_predict.predict(model_name_or_path=metricx_model, input_data=metricx_data, batch_size=4, device="cuda")
    avg_metricx = sum(metricx_scores) / len(metricx_scores)

    # 2. Append results to the list
    all_results.append({
        "System": system,
        "Language Pair": langpair,
        "SacreBLEU": bleu_score,
        "chrF++": chrf_score,
        "COMET": comet_score,
        "MetricX": avg_metricx
    })

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Metrics to display
metrics = ["SacreBLEU", "chrF++", "COMET", "MetricX"]

# Print dynamic tables
for metric in metrics:
    print(f"\n--- {metric} ---")
    
    # Pivot handles the layout automatically
    # Rows = all unique systems found in filenames
    # Columns = all unique language pairs found in filenames
    table = df.pivot(index="System", columns="Language Pair", values=metric)
    
    # Optional: Sort index so Apertium is usually first (alphabetical)
    table = table.sort_index()
    
    # Format output
    if metric == "MetricX":
        print(table.applymap(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"))
    else:
        print(table.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"))