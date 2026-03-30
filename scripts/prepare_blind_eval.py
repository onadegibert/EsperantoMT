import json
import random
from pathlib import Path
import pandas as pd


model_files = {
    "modelA": "data/chrf_scores/eo-es.marian.jsonl",
    "modelB": "data/chrf_scores/eo-es.nllb-200-3.3B.jsonl",
    "modelC": "data/chrf_scores/eo-es.llama_ft.jsonl"
}


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_ranking_csv(
    model_files: dict,
    out_csv: str = "ranking_blind.csv",
    n_samples: int = 100,
    seed: int = 42,
    include_reference: bool = True,
):
    random.seed(seed)

    # Load all models
    models = {}
    lengths = set()
    for model_name, fp in model_files.items():
        models[model_name] = load_jsonl(fp)
        lengths.add(len(models[model_name]))

    if len(lengths) != 1:
        raise ValueError(f"Models have different number of lines: {lengths}")

    n_total = lengths.pop()
    if n_samples > n_total:
        raise ValueError(f"Requested n_samples={n_samples} but only {n_total} lines exist.")

    # Sample indices (same for all models)
    idxs = random.sample(range(n_total), n_samples)

    rows = []
    model_names = list(model_files.keys())

    for rank_id, i in enumerate(idxs, start=1):
        # Pull entries for this index from each model
        entries = {m: models[m][i] for m in model_names}

        # Optional sanity checks: ensure same source/reference across models
        src0 = entries[model_names[0]].get("source")
        ref0 = entries[model_names[0]].get("reference")
        for m in model_names[1:]:
            if entries[m].get("source") != src0:
                raise ValueError(f"Source mismatch at line {i} between {model_names[0]} and {m}")
            if include_reference and entries[m].get("reference") != ref0:
                raise ValueError(f"Reference mismatch at line {i} between {model_names[0]} and {m}")

        # Build the three translation candidates and shuffle their order per row
        candidates = [
            (m, entries[m].get("hypothesis", "")) for m in model_names
        ]
        random.shuffle(candidates)

        # Map to T1/T2/T3
        (m1, t1), (m2, t2), (m3, t3) = candidates

        row = {
            "id": rank_id,
            "source": src0,
            # reference optional
            "reference": ref0 if include_reference else "",
            "T1": t1,
            "T2": t2,
            "T3": t3,

            # columns for your dad to fill:
            "millor_(1-3)": "",
            "pitjor_(1-3)": "",
            "comentari": "",

            # hidden columns (YOU hide these in Excel):
            "T1_model": m1,
            "T2_model": m2,
            "T3_model": m3,
            "orig_line_index": i,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Order columns nicely
    cols = [
        "id", "source"
    ] + (["reference"] if include_reference else []) + [
        "T1", "T2", "T3",
        "millor_(1-3)", "pitjor_(1-3)", "comentari",
        "T1_model", "T2_model", "T3_model", "orig_line_index"
    ]
    df = df[cols]

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {out_csv}")
    print("Tip: Hide columns T1_model, T2_model, T3_model, orig_line_index (and reference if you want).")


if __name__ == "__main__":
    prepare_ranking_csv(
        model_files=model_files,
        out_csv="ranking_blind.csv",
        n_samples=100,
        seed=42,
        include_reference=True,   # posa False si NO vols ensenyar la reference
    )
