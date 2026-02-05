from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path("/content/bert-squad-project/Projet")
DEV_HF = BASE_DIR / "Data" / "processed" / "dev_squad_hf"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128
SAMPLE_SIZE = 5000  # prends moins si c'est lent

# -----------------------------
# LOAD
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
ds = load_from_disk(str(DEV_HF))
ds_small = ds.shuffle(seed=42).select(range(min(SAMPLE_SIZE, len(ds))))

# -----------------------------
# 1) Longueur Q + Contexte (tokens)
# -----------------------------
lens = [
    len(tokenizer(q, c, truncation=False)["input_ids"])
    for q, c in zip(ds_small["question"], ds_small["context"])
]

over_384 = sum(l > MAX_LENGTH for l in lens)

print(f"Exemples > {MAX_LENGTH} tokens : {over_384}/{len(lens)} ({over_384/len(lens)*100:.2f}%)")
print("p50/p90/p95/max =",
      np.percentile(lens, 50),
      np.percentile(lens, 90),
      np.percentile(lens, 95),
      max(lens))

# Histogramme longueur Q+CTX + ligne seuil
plt.figure()
plt.hist(lens, bins=40)
plt.axvline(MAX_LENGTH, linestyle="--")
plt.title(f"Longueur Question+Contexte (tokens BERT) - DEV (seuil={MAX_LENGTH})")
plt.xlabel("Nombre de tokens (Q+CTX)")
plt.ylabel("Nombre d'exemples")
plt.savefig(str(FIG_DIR / "longueur_q_ctx_tokens_dev.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# -----------------------------
# 2) Fenêtrage (sliding window) pour exemples longs
# -----------------------------
windows_count = []

for q, c in zip(ds_small["question"], ds_small["context"]):
    total_len = len(tokenizer(q, c, truncation=False)["input_ids"])
    if total_len > MAX_LENGTH:
        tok = tokenizer(
            q.lstrip(),
            c,
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
        )
        windows_count.append(len(tok["input_ids"]))

print("\n--- Fenêtrage (stride) ---")
print(f"Nb exemples longs (>{MAX_LENGTH}):", len(windows_count))
if windows_count:
    print("Moyenne fenêtres par exemple long:", sum(windows_count) / len(windows_count))
    counts = Counter(windows_count)
    print("Répartition fenêtres:", dict(sorted(counts.items())))
else:
    print("Aucun exemple long dans cet échantillon.")

# Histogramme fenêtres par exemple long
if windows_count:
    plt.figure()
    plt.hist(windows_count, bins=range(1, max(windows_count) + 2), align="left")
    plt.title(f"Nombre de fenêtres générées par exemple (>{MAX_LENGTH} tokens) - stride={DOC_STRIDE}")
    plt.xlabel("Fenêtres (features) par exemple")
    plt.ylabel("Nombre d'exemples longs")
    plt.savefig(str(FIG_DIR / "fenetres_par_exemple_long_dev.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

print(f"\n✅ Figures sauvegardées dans : {FIG_DIR}")
print(" - longueur_q_ctx_tokens_dev.png")
print(" - fenetres_par_exemple_long_dev.png")