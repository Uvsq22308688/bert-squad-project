from datasets import load_from_disk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================
BASE_DIR = Path("/content/bert-squad-project")
DATA_DIR = BASE_DIR / "Projet" / "Data" / "processed" / "dev_squad_hf"
OUT_DIR  = BASE_DIR / "Projet" / "analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = OUT_DIR / "context_length_distribution_dev.png"

# =========================
# LOAD DATA
# =========================
print("📂 Loading dataset from:", DATA_DIR)
ds = load_from_disk(str(DATA_DIR))

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True
)

# Subsample for speed
ds_small = ds.shuffle(seed=42).select(
    range(min(5000, len(ds)))
)

# =========================
# COMPUTE LENGTHS
# =========================
ctx_token_lengths = [
    len(tokenizer(ctx, truncation=False)["input_ids"])
    for ctx in ds_small["context"]
]

# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.hist(ctx_token_lengths, bins=30)
plt.title("Distribution of context length (BERT tokens) – DEV")
plt.xlabel("Number of BERT tokens")
plt.ylabel("Number of contexts")
plt.tight_layout()

plt.savefig(FIG_PATH)
plt.close()

# =========================
# STATS
# =========================
stats = {
    "min": min(ctx_token_lengths),
    "mean": float(np.mean(ctx_token_lengths)),
    "p50": float(np.percentile(ctx_token_lengths, 50)),
    "p90": float(np.percentile(ctx_token_lengths, 90)),
    "p95": float(np.percentile(ctx_token_lengths, 95)),
    "max": max(ctx_token_lengths),
}

print("📊 Stats:", stats)
print("🖼️ Figure saved to:", FIG_PATH)
