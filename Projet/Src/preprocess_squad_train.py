"""
Prétraitement des Données - SQuAD TRAIN
- Charge train-v1.1.json (brut)
- Aplati en format HF compatible QA
- Vérifie l'alignement answer_start
- Sauvegarde en Hugging Face Dataset
"""

import json
from pathlib import Path
from datasets import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "Data" / "train-v1.1.json"
OUT_PATH = BASE_DIR / "Data" / "processed" / "train_squad_hf"

def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_squad(squad_json):
    records = []
    for article in squad_json["data"]:
        title = article.get("title", "")
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qid = qa["id"]
                question = qa["question"].strip()
                for ans in qa["answers"]:
                    records.append({
                        "id": qid,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": {
                            "text": [ans["text"]],
                            "answer_start": [ans["answer_start"]],
                        },
                    })
    return records

def check_alignment(records):
    errors = 0
    for r in records[:2000]:  # petit check rapide (peut enlever la limite)
        c = r["context"]
        start = r["answers"]["answer_start"][0]
        text = r["answers"]["text"][0]
        if c[start:start + len(text)] != text:
            errors += 1
    print(f"Alignment errors (sample): {errors}/2000")

def main():
    print("Loading:", RAW_PATH)
    squad = load_data(RAW_PATH)

    records = flatten_squad(squad)
    print("Rows:", len(records))

    check_alignment(records)

    ds = Dataset.from_list(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUT_PATH))
    print("Saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
