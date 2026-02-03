"""
Collecte et Prétraitement des Données - SQuAD DEV
UVSQ - M2 Datascale
"""
import json
from datasets import Dataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "dev-v1.1.json"
OUT_PATH = BASE_DIR / "data" / "processed" / "dev_squad_hf"


def load_dev_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_squad(dev_json):
    records = []
    for article in dev_json["data"]:
        title = article.get("title", "")
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qid = qa["id"]
                question = qa["question"]
                for ans in qa["answers"]:
                    records.append({
                        "id": qid,
                        "title": title,
                        "context": context,
                        "question": question.strip(),
                        "answers": {
                            "text": [ans["text"]],
                            "answer_start": [ans["answer_start"]]
                        }
                    })
    return records

def check_alignment(records):
    errors = 0
    for r in records:
        c = r["context"]
        start = r["answers"]["answer_start"][0]
        text = r["answers"]["text"][0]
        if c[start:start+len(text)] != text:
            errors += 1
    print(f"Alignment errors: {errors}/{len(records)}")

def main():
    dev_json = load_dev_data(RAW_PATH)
    records = flatten_squad(dev_json)
    check_alignment(records)

    dev_ds = Dataset.from_list(records)
    dev_ds.save_to_disk(OUT_PATH)

    print("DEV dataset preprocessed and saved ✔")

if __name__ == "__main__":
    main()
