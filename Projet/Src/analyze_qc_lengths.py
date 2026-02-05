from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path

def find_dataset_dir(name="dev_squad_hf", search_root="/content/bert-squad-project/Projet/Data/processed"):
    root = Path(search_root)
    hits = [p for p in root.rglob(name) if p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"❌ Dataset '{name}' introuvable dans {search_root}")
    return hits[0]

def main():
    # 🔍 Trouver automatiquement le dataset
    DATA_DIR = find_dataset_dir("dev_squad_hf", "/content/bert-squad-project/Projet/Data/processed")

    # Remonter jusqu'au dossier Projet
    BASE_DIR = DATA_DIR
    for _ in range(4):  # dev_squad_hf → Traitée → Données → Projet
        BASE_DIR = BASE_DIR.parent

    OUT_DIR = BASE_DIR / "Projet" / "analysis"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE = OUT_DIR / "qc_token_length_stats.txt"

    print("📂 Dataset :", DATA_DIR)
    print("📁 Output  :", OUT_FILE)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    ds = load_from_disk(str(DATA_DIR))

    ds_small = ds.shuffle(seed=42).select(range(min(5000, len(ds))))

    # 🔢 Longueur question + contexte
    lens = [
        len(tokenizer(q, c, truncation=False)["input_ids"])
        for q, c in zip(ds_small["question"], ds_small["context"])
    ]

    over_384 = sum(l > 384 for l in lens)

    stats = {
        "total_examples": len(lens),
        "over_384": over_384,
        "percent_over_384": over_384 / len(lens) * 100,
        "p50": np.percentile(lens, 50),
        "p90": np.percentile(lens, 90),
        "p95": np.percentile(lens, 95),
        "max": max(lens),
    }

    # 💾 Sauvegarde texte
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print("✅ Résultats sauvegardés")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()