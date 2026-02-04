import pandas as pd
import os, json

df = pd.read_csv("data/partial_results.csv")
MX_OUT = "data/metricx_scores"

# Add MetricX scores
mx_list = []
for file in os.listdir(MX_OUT):
    scores = [json.loads(l)["prediction"] for l in open(os.path.join(MX_OUT, file))]
    mx_list.append({
        "System": file.split(".")[1],
        "LP": file.split(".")[0],
        "MetricX": sum(scores) / len(scores)
    })

df = df.merge(pd.DataFrame(mx_list), on=["System", "LP"], how="left")
df["COMET"] = df["COMET"] * 100

# Print Tables
for m in ["BLEU", "chrF", "COMET", "MetricX"]:
    print(f"\n--- {m} Table ---")
    print(df.pivot(index="System", columns="LP", values=m).round(2))
