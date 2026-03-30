import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_ranking(csv_path):
    df = pd.read_csv(csv_path, delimiter="\t")

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

    results = {}
    total = len(df_valid)

    print(f"Total rows in CSV: {len(df)}")
    print(f"Valid annotated rows: {total}\n")

    for _, row in df_valid.iterrows():
        best_col = f"T{row['millor_(1-3)']}_model"
        worst_col = f"T{row['pitjor_(1-3)']}_model"

        best_model = row[best_col]
        worst_model = row[worst_col]

        for m in [best_model, worst_model]:
            if m not in results:
                results[m] = {"best": 0, "worst": 0}

        results[best_model]["best"] += 1
        results[worst_model]["worst"] += 1

    summary = []

    for model, counts in results.items():
        best = counts["best"]
        worst = counts["worst"]
        summary.append({
            "model": model,
            "best": best,
            "worst": worst,
            "best_minus_worst": best - worst,
            "win_rate": round(best / total, 3) if total > 0 else 0
        })

    #summary_df = pd.DataFrame(summary)
    summary_df = pd.DataFrame(summary).sort_values(by="best", ascending=False)
    # Force final order: nllb, llama, marian
    #final_order = ["nllb", "llama", "marian"]
    #summary_df["__order"] = pd.Categorical(summary_df["model"], categories=final_order, ordered=True)
    #summary_df = summary_df.sort_values("__order").drop(columns="__order")

    print(summary_df.to_string(index=False))
    return(summary_df, total)

def plot_win_rates(
    summary_df,
    total_sentences,
    name_map,
    color_map,
    out_path="figs/win_rates.papa.png",
    as_percent=True
):
    """
    summary_df: DataFrame with columns ['model', 'best'] (at minimum)
    total_sentences: number of valid annotated rows
    name_map: dict like {'modelA': 'Marian', 'modelB': 'NLLB-200 3.3B', ...}
    color_map: dict like {'modelA': '659157', 'modelB': '987284', 'modelC': 'DE9E36'}
               (hex strings with or without '#')
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = summary_df.copy()

    # Compute win rate
    df["win_rate"] = df["best"] / float(total_sentences)
    if as_percent:
        df["win_rate"] = 100.0 * df["win_rate"]

    # Map to display names + colors
    df["display_name"] = df["model"].map(name_map).fillna(df["model"])

    def norm_hex(h):
        h = str(h).strip()
        return h if h.startswith("#") else f"#{h}"

    df["color"] = df["model"].map(color_map).apply(lambda x: norm_hex(x) if x is not None else None)

    x = df["display_name"].tolist()
    y = df["win_rate"].tolist()
    colors = df["color"].tolist()

    # Plot
    plt.figure(figsize=(3,5))
    bars = plt.bar(x, y, color=colors, edgecolor="black", linewidth=0.8)
    ylabel = "Win rate (%)" if as_percent else "Win rate (proportion)"
    plt.ylabel(ylabel)
    #plt.title("Human ranking: win rate by model")
    plt.ylim(0,60)
    plt.xticks(rotation=45, ha="right")
    #plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.margins(y=0.12)

    # Add value labels on bars
    for b, val in zip(bars, y):
        label = f"{val:.1f}%" if as_percent else f"{val:.3f}"
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    summary, counts = analyze_ranking("data/papa.tsv")
    name_map = {
        "marian": "Transformer-base",
        "nllb": "NLLB-200-3.3B",
        "llama": "Llama-3.1-8B-FT",
    }

    # Your requested colors
    color_map = {
        "marian": "659157",
        "nllb": "987284",
        "llama": "DE9E36",
    }
    name_map = {
        "modelA": "Transformer-base",
        "modelB": "NLLB-200-3.3B",
        "modelC": "Llama-3.1-8B-FT",
    }

    # Your requested colors
    color_map = {
        "modelA": "659157",
        "modelB": "987284",
        "modelC": "DE9E36",
    }
    


    plot_win_rates(summary, counts, name_map, color_map)
