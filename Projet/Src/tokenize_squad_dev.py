"""
Tokenisation SQuAD DEV (v1.1) - préparation des features pour QA extractif
- Charge le dataset preprocessé (save_to_disk)
- Tokenise question+context avec overflow/stride
- Calcule start_positions / end_positions à partir de answer_start + offset_mapping
- Sauvegarde le dataset tokenisé sur disque

Exécution (depuis le dossier Projet ou Src) :
python Src/tokenize_squad_dev.py
"""

from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

# -----------------------------
# CONFIG
# -----------------------------
MODEL_CHECKPOINT = "bert-base-uncased"  # tu pourras changer: roberta-base, distilbert-base-uncased, etc.
MAX_LENGTH = 384
DOC_STRIDE = 128

# Important: on rend le script robuste au "où je le lance"
BASE_DIR = Path(__file__).resolve().parent.parent  # .../Projet
IN_PATH = BASE_DIR / "Data" / "processed" / "dev_squad_hf"
OUT_PATH = BASE_DIR / "Data" / "processed" / "dev_squad_tokenized" / MODEL_CHECKPOINT.replace("/", "_")

# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"Loading dataset from: {IN_PATH}")
    dev_ds = load_from_disk(str(IN_PATH))
    print(dev_ds)

    print(f"Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    # Fonction de préparation
    def prepare_features(examples):
        # Important: on strip la question, pas le contexte
        questions = [q.lstrip() for q in examples["question"]]

        tokenized = tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # overflow_to_sample_mapping: pour chaque "feature", à quel exemple original ça correspond
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []
        example_ids = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sample_index = sample_mapping[i]
            example_ids.append(examples["id"][sample_index])

            answers = examples["answers"][sample_index]
            # SQuAD v1.1: au moins une réponse
            answer_start_char = answers["answer_start"][0]
            answer_text = answers["text"][0]
            answer_end_char = answer_start_char + len(answer_text)

            # sequence_ids: indique quels tokens viennent de la question(0) vs contexte(1) vs special(None)
            sequence_ids = tokenized.sequence_ids(i)

            # Trouver les limites tokens du contexte dans cette feature
            # (first token with sequence_id==1) ... (last token with sequence_id==1)
            context_start = None
            context_end = None
            for idx, sid in enumerate(sequence_ids):
                if sid == 1 and context_start is None:
                    context_start = idx
                if sid == 1:
                    context_end = idx

            # Si pas de contexte (très rare), fallback
            if context_start is None or context_end is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            # Vérifier si la réponse est entièrement dans la fenêtre courante
            # offsets[idx] = (start_char, end_char) dans le texte original
            if not (offsets[context_start][0] <= answer_start_char and offsets[context_end][1] >= answer_end_char):
                # réponse hors fenêtre -> CLS
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Chercher start token
                token_start = context_start
                while token_start <= context_end and offsets[token_start][0] <= answer_start_char:
                    token_start += 1
                start_pos = token_start - 1

                # Chercher end token
                token_end = context_end
                while token_end >= context_start and offsets[token_end][1] >= answer_end_char:
                    token_end -= 1
                end_pos = token_end + 1

                start_positions.append(start_pos)
                end_positions.append(end_pos)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized["example_id"] = example_ids

        return tokenized

    print("Tokenizing + building start/end positions...")
    tokenized_dev = dev_ds.map(
        prepare_features,
        batched=True,
        remove_columns=dev_ds.column_names,
        desc="Tokenizing DEV",
    )

    print(tokenized_dev)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving tokenized dataset to: {OUT_PATH}")
    tokenized_dev.save_to_disk(str(OUT_PATH))

    # Petit contrôle: combien de features et aperçu colonnes
    print("Done ✔")
    print("Columns:", tokenized_dev.column_names)
    print("Example tokenized row:", {k: tokenized_dev[0][k] for k in ["example_id", "start_positions", "end_positions"]})

if __name__ == "__main__":
    main()
