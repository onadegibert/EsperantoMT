import os, json, evaluate
import pandas as pd
from datasets import load_dataset
from comet import download_model, load_from_checkpoint

token = open(".hf_token", "r").read().strip()

# Initialize
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

PRED_DIR = "data/predictions/"
MX_IN_DIR = "data/metricx_inputs/"
os.makedirs(MX_IN_DIR, exist_ok=True)

lang_codes = {"en": "eng_Latn", "eo": "epo_Latn", "ca": "cat_Latn", "es": "spa_Latn"}

results = []

for pred_file in os.listdir(PRED_DIR):
    lp = pred_file.split(".")[0]
    src_iso, tgt_iso = lp.split("-")

    src_ds = load_dataset(
        "openlanguagedata/flores_plus",
        lang_codes[src_iso],
        split="devtest",
        token=token,
    )
    tgt_ds = load_dataset(
        "openlanguagedata/flores_plus",
        lang_codes[tgt_iso],
        split="devtest",
        token=token,
    )

    with open(os.path.join(PRED_DIR, pred_file), "r") as f:
        hyps_raw = [l.rstrip("\n") for l in f]  # keep internal spacing, just drop newline

    srcs = [x["text"] for x in src_ds]
    refs = [x["text"] for x in tgt_ds]

    # If apertium: count and strip asterisks
    is_apertium = "apertium" in pred_file.lower()
    unknown_words = 0

    if is_apertium:
        unknown_words = sum(h.count("*") for h in hyps_raw)
        hyps = [h.replace("*", "") for h in hyps_raw]
    else:
        hyps = [h.strip() for h in hyps_raw]  # original behavior

    # 1. Save JSONL for Step 2 (hypothesis written WITHOUT asterisks for apertium)
    with open(os.path.join(MX_IN_DIR, f"{pred_file}.jsonl"), "w") as f_out:
        for s, r, h in zip(srcs, refs, hyps):
            f_out.write(json.dumps({"source": s, "reference": r, "hypothesis": h}) + "\n")

    # 2. Compute Metrics (WITHOUT asterisks for apertium)
    b = bleu.compute(predictions=hyps, references=[[r] for r in refs])["score"]
    c = chrf.compute(predictions=hyps, references=[[r] for r in refs], word_order=2)["score"]
    cm = comet_model.predict(
        [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)],
        batch_size=16,
        gpus=1,
    ).system_score

    results.append(
        {
            "System": pred_file.split(".")[1],
            "LP": lp,
            "BLEU": b,
            "chrF": c,
            "COMET": cm,
            "UnknownWords": unknown_words,
        }
    )

# Save partial results
pd.DataFrame(results).to_csv("data/partial_results.csv", index=False)

