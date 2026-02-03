import time
import re
import numpy as np
from collections import Counter
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# =====================
# PATHS 
# =====================
MODEL_DIR = "/kaggle/working/Projet/models/bert_squad_finetuned_bert"

# : question / context / answers
RAW_DEV_PATH = "/kaggle/working/Projet/Data/processed/train_squad_hf"

# =====================
# LOAD MODEL
# =====================
print("📥 Loading model/tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)

device_id = 0 if model.device.type == "cuda" else -1

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

# إذا كان DatasetDict خدي split ديال validation ولا dev
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

# =====================
# METRICS (SQuAD-style)
# =====================
def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)   # remove articles
    text = re.sub(r"[^\w\s]", "", text)          # remove punctuation
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

print("\n🚀 Starting FULL evaluation... (this may take time)\n")

t_global = time.time()

for idx, ex in enumerate(dev_ds):
    question = ex["question"]
    context = ex["context"]

    # : answers["text"] = [...]
    gold = ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else ""

    t0 = time.time()
    pred = qa({"question": question, "context": context})
    infer_time = time.time() - t0

    answer = pred["answer"]

    EMs.append(exact_match(answer, gold))
    F1s.append(f1(answer, gold))
    Ps.append(precision(answer, gold))
    Rs.append(recall(answer, gold))
    times.append(infer_time)

    # progress  5000
    if (idx + 1) % 5000 == 0:
        print(f"Progress: {idx+1}/{len(dev_ds)} done...")

total_eval_time = time.time() - t_global

# =====================
# RESULTS
# =====================
print("\n📊 FINAL RESULTS (FULL VALIDATION)")
print("-" * 55)
print(f"Exact Match (EM):   {np.mean(EMs)*100:.2f} %")
print(f"F1-score:           {np.mean(F1s)*100:.2f} %")
print(f"Precision:          {np.mean(Ps)*100:.2f} %")
print(f"Recall:             {np.mean(Rs)*100:.2f} %")
print(f"Avg inference time: {np.mean(times)*1000:.2f} ms/question")
print(f"Total evaluation time: {total_eval_time/60:.2f} minutes")
print("-" * 55)
print("✅ DONE")