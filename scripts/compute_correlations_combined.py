import pandas as pd
import json
import numpy as np
from scipy.stats import kendalltau


# ---------- 1. Load human TSV ----------
def load_human_annotations(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t")

    # Convert to string and strip spaces
    df["millor_(1-3)"] = df["millor_(1-3)"].astype(str).str.strip()
    df["pitjor_(1-3)"] = df["pitjor_(1-3)"].astype(str).str.strip()

    # Keep only rows where both are 1,2,3
    valid_mask = (
        df["millor_(1-3)"].isin(["1", "2", "3"]) &
        df["pitjor_(1-3)"].isin(["1", "2", "3"])
    )

    df_valid = df[valid_mask].copy()

    # Convert to int
    df_valid["millor_(1-3)"] = df_valid["millor_(1-3)"].astype(int)
    df_valid["pitjor_(1-3)"] = df_valid["pitjor_(1-3)"].astype(int)

    total = len(df_valid)

    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid annotated rows: {total}\n")

    return df_valid


# ---------- 2. Load metric scores ----------
def load_metric_jsonl(path, metric_key="chrf"):
    scores = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scores.append(obj[metric_key])
    return scores


# ---------- Helpers ----------
def get_human_order_for_row(row):
    """Return systems ordered best->middle->worst for this row."""
    idx_best = row["millor_(1-3)"] - 1
    idx_worst = row["pitjor_(1-3)"] - 1

    idx_middle = [0, 1, 2]
    idx_middle.remove(idx_best)
    idx_middle.remove(idx_worst)
    idx_middle = idx_middle[0]

    systems = [row["T1_model"], row["T2_model"], row["T3_model"]]
    human_order = [systems[idx_best], systems[idx_middle], systems[idx_worst]]
    return human_order


def kendall_tau_from_orders(human_order, metric_order):
    """
    Compute Kendall tau between two permutations of the same 3 items.
    We map items to ranks and run kendalltau on rank vectors.
    """
    items = human_order[:]  # same set
    human_rank = {m: i for i, m in enumerate(human_order)}   # 0 best ... 2 worst
    metric_rank = {m: i for i, m in enumerate(metric_order)}

    x = [human_rank[m] for m in items]
    y = [metric_rank[m] for m in items]
    tau, _ = kendalltau(x, y)
    # For 3 items tau should be in {-1, -1/3, 1/3, 1} depending on ties; here no ties in orders.
    return tau


# ---------- 3A. Kendall τ over model rankings ----------
def kendall_over_model_rankings(df, metric_scores_by_model, direction):
    """
    For each row: compare human ranking (best>mid>worst) vs metric-induced ranking (sort by scores).
    Returns:
      - mean_tau: average tau over rows
      - pooled_tau: tau computed over all pairwise comparisons pooled (equivalent to accuracy mapping, but as tau)
      - per_row_taus: list of taus
    """
    taus = []
    # For pooled tau, we collect pairwise labels/signs similarly to your original code
    pooled_human = []
    pooled_metric = []

    for _, row in df.iterrows():
        idx = int(row["orig_line_index"])
        human_order = get_human_order_for_row(row)

        # metric order by descending score
        scores = {m: metric_scores_by_model[m][idx] for m in human_order}
        reverse = (direction == "higher")

        metric_order = sorted(
            human_order,
            key=lambda m: scores[m],
            reverse=reverse
        )

        tau = kendall_tau_from_orders(human_order, metric_order)
        taus.append(tau)

        # pooled pairwise (3 comparisons)
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            better = human_order[i]
            worse = human_order[j]
            pooled_human.append(1)  # better>worse always 1
            diff = metric_scores_by_model[better][idx] - metric_scores_by_model[worse][idx]
            pooled_metric.append(1 if diff > 0 else -1 if diff < 0 else 0)

    mean_tau = float(np.nanmean(taus))
    pooled_tau, pooled_p = kendalltau(pooled_human, pooled_metric)
    return mean_tau, pooled_tau, pooled_p, taus


# ---------- 3B. Kendall τ over model scores ----------
def kendall_over_model_scores(df, metric_scores_by_model, direction):
    """
    Compute Kendall τ between human ranks and metric *scores* (not metric ranks).
    We do two variants:
      1) pooled over all (row, system) points: tau(human_rank, metric_score)
      2) average of per-row tau: tau([0,1,2], scores aligned to human_order)
    """
    pooled_human_ranks = []
    pooled_metric_scores = []
    per_row_taus = []

    for _, row in df.iterrows():
        idx = int(row["orig_line_index"])
        human_order = get_human_order_for_row(row)

        # human ranks: 0 best, 1 mid, 2 worst
        ranks = [0, 1, 2]
        scores = [metric_scores_by_model[m][idx] for m in human_order]

        if direction == "lower":
            ranks = [2,1,0]
            scores = [-s for s in scores]

        # per-row tau between ranks and scores
        tau_row, _ = kendalltau(ranks, scores)
        per_row_taus.append(tau_row)

        # pooled
        pooled_human_ranks.extend(ranks)
        pooled_metric_scores.extend(scores)

    pooled_tau, pooled_p = kendalltau(pooled_human_ranks, pooled_metric_scores)
    mean_tau = float(np.nanmean(per_row_taus))
    return mean_tau, pooled_tau, pooled_p, per_row_taus


# ---------- 3C. Pairwise Accuracy ----------
def pairwise_accuracy(df, metric_scores_by_model, direction):
    """
    Accuracy over the 3 pairwise comparisons per row:
      (best vs mid), (best vs worst), (mid vs worst)
    Metric agrees if score(better) > score(worse). Ties count as 0.5 by default (configurable).
    """
    correct = 0.0
    total = 0

    for _, row in df.iterrows():
        idx = int(row["orig_line_index"])
        human_order = get_human_order_for_row(row)

        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            better = human_order[i]
            worse = human_order[j]
            sb = metric_scores_by_model[better][idx]
            sw = metric_scores_by_model[worse][idx]

            total += 1
            
            if direction == "higher":
                if sb > sw:
                    correct += 1.0
                elif sb == sw:
                    correct += 0.5

            else:  # lower is better

                if sb < sw:
                    correct += 1.0
                elif sb == sw:
                    correct += 0.5

    return correct / total if total > 0 else float("nan")


# ---------- Example Usage ----------
if __name__ == "__main__":

    # Load human annotations
    df_es_eo = load_human_annotations("data/papa.tsv")
    df_eo_es=load_human_annotations("data/ona_2.tsv")
    df = pd.concat([df_es_eo, df_eo_es])

    metric_direction = {
        "chrF++": "higher",
        "BLEU": "higher",
        "Comet": "higher",
        "MetricX": "lower"
    }


    # Load metric scores
    metric_scores = {
        "modelA": load_metric_jsonl("data/chrf_scores/es-eo.marian.jsonl", "chrf"),
        "modelB": load_metric_jsonl("data/chrf_scores/es-eo.nllb-200-3.3B.jsonl", "chrf"),
        "modelC": load_metric_jsonl("data/chrf_scores/es-eo.llama_ft.jsonl", "chrf"),
    }

    print("ChrF++")
    direction=metric_direction["chrF++"]
    # --- Kendall τ over model rankings (human order vs metric-induced order) ---
    mean_tau_rank, pooled_tau_rank, pooled_p_rank, _ = kendall_over_model_rankings(df, metric_scores, direction)
    print("Kendall τ over model RANKINGS")
    print(f"  mean(per-sample τ) = {mean_tau_rank:.4f}")
    print(f"  pooled τ           = {pooled_tau_rank:.4f}")
    print(f"  pooled p-value     = {pooled_p_rank:.6g}\n")

    # --- Kendall τ over model scores (human ranks vs metric scores) ---
    mean_tau_score, pooled_tau_score, pooled_p_score, _ = kendall_over_model_scores(df, metric_scores, direction)
    print("Kendall τ over model SCORES")
    print(f"  mean(per-sample τ) = {mean_tau_score:.4f}")
    print(f"  pooled τ           = {pooled_tau_score:.4f}")
    print(f"  pooled p-value     = {pooled_p_score:.6g}\n")

    # --- Pairwise accuracy ---
    acc = pairwise_accuracy(df, metric_scores, direction)
    print("Pairwise Accuracy")
    print(f"  accuracy = {acc:.4f}")

    metric_scores = {
        "modelA": load_metric_jsonl("data/bleu_scores/es-eo.marian.jsonl", "bleu"),
        "modelB": load_metric_jsonl("data/bleu_scores/es-eo.nllb-200-3.3B.jsonl", "bleu"),
        "modelC": load_metric_jsonl("data/bleu_scores/es-eo.llama_ft.jsonl", "bleu"),
    }

    print("\n\nBLEU")
    direction=metric_direction["BLEU"]
    # --- Kendall τ over model rankings (human order vs metric-induced order) ---
    mean_tau_rank, pooled_tau_rank, pooled_p_rank, _ = kendall_over_model_rankings(df, metric_scores, direction)
    print("Kendall τ over model RANKINGS")
    print(f"  mean(per-sample τ) = {mean_tau_rank:.4f}")
    print(f"  pooled τ           = {pooled_tau_rank:.4f}")
    print(f"  pooled p-value     = {pooled_p_rank:.6g}\n")

    # --- Kendall τ over model scores (human ranks vs metric scores) ---
    mean_tau_score, pooled_tau_score, pooled_p_score, _ = kendall_over_model_scores(df, metric_scores, direction)
    print("Kendall τ over model SCORES")
    print(f"  mean(per-sample τ) = {mean_tau_score:.4f}")
    print(f"  pooled τ           = {pooled_tau_score:.4f}")
    print(f"  pooled p-value     = {pooled_p_score:.6g}\n")

    # --- Pairwise accuracy ---
    acc = pairwise_accuracy(df, metric_scores, direction)
    print("Pairwise Accuracy")
    print(f"  accuracy = {acc:.4f}")
    

    metric_scores = {
        "modelA": load_metric_jsonl("data/comet_scores/es-eo.marian.jsonl", "comet"),
        "modelB": load_metric_jsonl("data/comet_scores/es-eo.nllb-200-3.3B.jsonl", "comet"),
        "modelC": load_metric_jsonl("data/comet_scores/es-eo.llama_ft.jsonl", "comet"),
    }

    print("\n\nComet")
    direction=metric_direction["Comet"]

    # --- Kendall τ over model rankings (human order vs metric-induced order) ---
    mean_tau_rank, pooled_tau_rank, pooled_p_rank, _ = kendall_over_model_rankings(df, metric_scores, direction)
    print("Kendall τ over model RANKINGS")
    print(f"  mean(per-sample τ) = {mean_tau_rank:.4f}")
    print(f"  pooled τ           = {pooled_tau_rank:.4f}")
    print(f"  pooled p-value     = {pooled_p_rank:.6g}\n")

    # --- Kendall τ over model scores (human ranks vs metric scores) ---
    mean_tau_score, pooled_tau_score, pooled_p_score, _ = kendall_over_model_scores(df, metric_scores, direction)
    print("Kendall τ over model SCORES")
    print(f"  mean(per-sample τ) = {mean_tau_score:.4f}")
    print(f"  pooled τ           = {pooled_tau_score:.4f}")
    print(f"  pooled p-value     = {pooled_p_score:.6g}\n")

    # --- Pairwise accuracy ---
    acc = pairwise_accuracy(df, metric_scores, direction)
    print("Pairwise Accuracy")
    print(f"  accuracy = {acc:.4f}")

    metric_scores = {
        "modelA": load_metric_jsonl("data/metricx_scores/es-eo.marian.jsonl", "prediction"),
        "modelB": load_metric_jsonl("data/metricx_scores/es-eo.nllb-200-3.3B.jsonl", "prediction"),
        "modelC": load_metric_jsonl("data/metricx_scores/es-eo.llama_ft.jsonl", "prediction"),
    }

    print("\n\nMetricX")
    direction=metric_direction["MetricX"]

    # --- Kendall τ over model rankings (human order vs metric-induced order) ---
    mean_tau_rank, pooled_tau_rank, pooled_p_rank, _ = kendall_over_model_rankings(df, metric_scores, direction)
    print("Kendall τ over model RANKINGS")
    print(f"  mean(per-sample τ) = {mean_tau_rank:.4f}")
    print(f"  pooled τ           = {pooled_tau_rank:.4f}")
    print(f"  pooled p-value     = {pooled_p_rank:.6g}\n")

    # --- Kendall τ over model scores (human ranks vs metric scores) ---
    mean_tau_score, pooled_tau_score, pooled_p_score, _ = kendall_over_model_scores(df, metric_scores, direction)
    print("Kendall τ over model SCORES")
    print(f"  mean(per-sample τ) = {mean_tau_score:.4f}")
    print(f"  pooled τ           = {pooled_tau_score:.4f}")
    print(f"  pooled p-value     = {pooled_p_score:.6g}\n")

    # --- Pairwise accuracy ---
    acc = pairwise_accuracy(df, metric_scores, direction)
    print("Pairwise Accuracy")
    print(f"  accuracy = {acc:.4f}")