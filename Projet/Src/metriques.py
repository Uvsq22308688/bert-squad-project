import time
import re
import os
import json
import csv
import numpy as np
from collections import Counter
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# =====================
# PATHS (Colab)
# =====================
MODEL_DIR = "/content/bert-squad-project/Projet/models/bert_squad_finetuned_bert"
RAW_DEV_PATH = "/content/bert-squad-project/Projet/Data/processed/dev_squad_hf"

# =====================
# OUTPUT FILES
# =====================
RESULTS_DIR = "/content/bert-squad-project/Projet/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(RESULTS_DIR, "metrics_summary.txt")
DETAILS_PATH = os.path.join(RESULTS_DIR, "metrics_details.csv")
JSON_PATH    = os.path.join(RESULTS_DIR, "metrics_summary.json")

# =====================
# LOAD MODEL (GPU)
# =====================
print("📥 Loading model/tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)

device_id = 0 if torch.cuda.is_available() else -1
if device_id == 0:
    model = model.to("cuda")
    print("✅ Using GPU")
else:
    print("⚠️ Using CPU")

qa = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=device_id
)

# =====================
# LOAD RAW VALIDATION DATASET
# =====================
print("📥 Loading RAW validation dataset...")
ds = load_from_disk(RAW_DEV_PATH)

if hasattr(ds, "keys"):
    if "validation" in ds:
        dev_ds = ds["validation"]
    elif "dev" in ds:
        dev_ds = ds["dev"]
    else:
        dev_ds = list(ds.values())[0]
else:
    dev_ds = ds

print("✅ Dataset loaded.")
print("Number of examples:", len(dev_ds))
print("Columns:", dev_ds.column_names)

# (Optional) Quick test on a smaller subset
# dev_ds = dev_ds.select(range(500))

# =====================
# METRICS (SQuAD-style)
# =====================
def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

def precision(pred, gold):
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    if len(pred_toks) == 0:
        return 0.0
    return sum(common.values()) / len(pred_toks)

def recall(pred, gold):
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    if len(gold_toks) == 0:
        return 0.0
    return sum(common.values()) / len(gold_toks)

def f1(pred, gold):
    p = precision(pred, gold)
    r = recall(pred, gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

# =====================
# FULL EVALUATION LOOP
# =====================
EMs, F1s, Ps, Rs, times = [], [], [], [], []
details = []  # store per-example results

print("\n🚀 Starting FULL evaluation... (this may take time)\n")
t_global = time.time()

for idx, ex in enumerate(dev_ds):
    question = ex["question"]
    context = ex["context"]

    gold = ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else ""

    t0 = time.time()
    pred = qa(question=question, context=context)
    infer_time = time.time() - t0

    answer = pred["answer"]

    em_i = exact_match(answer, gold)
    f1_i = f1(answer, gold)
    p_i  = precision(answer, gold)
    r_i  = recall(answer, gold)

    EMs.append(em_i)
    F1s.append(f1_i)
    Ps.append(p_i)
    Rs.append(r_i)
    times.append(infer_time)

    details.append({
        "id": ex.get("id", idx),
        "question": question,
        "gold_answer": gold,
        "predicted_answer": answer,
        "exact_match": em_i,
        "f1": f1_i,
        "precision": p_i,
        "recall": r_i,
        "inference_time_sec": infer_time
    })

    if (idx + 1) % 5000 == 0:
        print(f"Progress: {idx+1}/{len(dev_ds)} done...")

total_eval_time = time.time() - t_global

# =====================
# RESULTS (PRINT)
# =====================
EM_mean = np.mean(EMs) * 100
F1_mean = np.mean(F1s) * 100
P_mean  = np.mean(Ps) * 100
R_mean  = np.mean(Rs) * 100
T_mean  = np.mean(times) * 1000  # ms/question

print("\n📊 FINAL RESULTS (FULL VALIDATION)")
print("-" * 55)
print(f"Exact Match (EM):   {EM_mean:.2f} %")
print(f"F1-score:           {F1_mean:.2f} %")
print(f"Precision:          {P_mean:.2f} %")
print(f"Recall:             {R_mean:.2f} %")
print(f"Avg inference time: {T_mean:.2f} ms/question")
print(f"Total evaluation time: {total_eval_time/60:.2f} minutes")
print("-" * 55)
print("✅ DONE")

# =====================
# SAVE RESULTS
# =====================

# TXT summary
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("FINAL EVALUATION RESULTS (SQuAD)\n")
    f.write("=" * 55 + "\n")
    f.write(f"Exact Match (EM):   {EM_mean:.2f} %\n")
    f.write(f"F1-score:           {F1_mean:.2f} %\n")
    f.write(f"Precision:          {P_mean:.2f} %\n")
    f.write(f"Recall:             {R_mean:.2f} %\n")
    f.write(f"Avg inference time: {T_mean:.2f} ms/question\n")
    f.write(f"Total eval time:    {total_eval_time/60:.2f} minutes\n")
    f.write(f"Num examples:       {len(dev_ds)}\n")

# JSON summary
summary_json = {
    "exact_match": EM_mean,
    "f1": F1_mean,
    "precision": P_mean,
    "recall": R_mean,
    "avg_inference_time_ms": T_mean,
    "total_eval_time_min": total_eval_time / 60,
    "num_examples": len(dev_ds),
    "model_dir": MODEL_DIR,
    "raw_dev_path": RAW_DEV_PATH
}
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=4)

# CSV details
if len(details) > 0:
    with open(DETAILS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)

print("\n💾 Results saved to:")
print(" -", SUMMARY_PATH)
print(" -", JSON_PATH)
print(" -", DETAILS_PATH)
