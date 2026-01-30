import os
os.environ["TRANSFORMERS_CACHE"] = ".hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = ".hf_cache"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate  # new library replacing load_metric
import torch
import sys


print("Available GPUs:",torch.cuda.device_count())

lang_codes = {"eng":"eng_Latn",
    "epo":"epo_Latn",
    "cat":"cat_Latn",
    "spa":"spa_Latn"
    }

langpair=sys.argv[1]
model_hf=sys.argv[2]

print(f"Evaluating {model_hf}...")

SRC_LANG, TGT_LANG =langpair.split("-")
SRC_LANG_CODE = lang_codes[SRC_LANG]
TGT_LANG_CODE = lang_codes[TGT_LANG]

print("Language pair: ", langpair)

modelname=model_hf.split("/")[1]

# Paths
model_dir = "models/{}".format(modelname)
src_file = "data/test.{}".format(SRC_LANG)
tgt_file = "data/test.{}".format(TGT_LANG)
hyp_file = "models/{}/{}.translation.out".format(modelname,langpair)
os.makedirs(model_dir, exist_ok=True)  # succeeds even if directory exists.

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_hf,
    src_lang=SRC_LANG_CODE,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_hf)
model.cuda()
model.eval()

# Load data
with open(src_file, "r", encoding="utf-8") as f:
    src_sentences = [line.strip() for line in f.readlines()]

with open(tgt_file, "r", encoding="utf-8") as f:
    tgt_sentences = [line.strip() for line in f.readlines()]

# Translate
batch_size = 16
generated_sentences = []

for i in range(0, len(src_sentences), batch_size):
    batch = src_sentences[i:i + batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # set target language for NLLB
    tokenizer.tgt_lang = TGT_LANG_CODE
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )
    batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_sentences.extend(batch_translations)

with open(hyp_file, "w", encoding="utf-8") as f:
    for line in generated_sentences:
        f.write(line + "\n")

# Evaluate with sacrebleu
# Load metrics
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

# Format references for evaluate (list of lists)
formatted_refs = [[t] for t in tgt_sentences]

# Compute SacreBLEU
bleu_results = bleu_metric.compute(
    predictions=generated_sentences, 
    references=formatted_refs
)

# Compute chrF++ (word_order=2 makes it chrF++)
chrf_results = chrf_metric.compute(
    predictions=generated_sentences, 
    references=formatted_refs, 
    word_order=2
)

print(f"SacreBLEU score: {bleu_results['score']:.2f}")
print(f"ChrF++ score: {chrf_results['score']:.2f}")

# Path for the results file
results_file = "models/{}/{}.results".format(modelname, langpair)

# Write results to file
with open(results_file, "w", encoding="utf-8") as f:
    f.write(f"Model: {model}\n")
    f.write(f"Language Pair: {langpair}\n")
    f.write(f"SacreBLEU: {bleu_results['score']:.2f}\n")
    f.write(f"chrF++: {chrf_results['score']:.2f}\n")

print(f"Results saved to {results_file}")
