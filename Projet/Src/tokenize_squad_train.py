"""
Tokenisation SQuAD TRAIN - création des features QA (PyTorch/Transformers)
"""

from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL_CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "Data" / "processed" / "train_squad_hf"
OUT_PATH = BASE_DIR / "Data" / "processed" / "train_squad_tokenized" / MODEL_CHECKPOINT.replace("/", "_")

def main():
    train_ds = load_from_disk(str(IN_PATH))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    def prepare_features(examples):
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

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            answer_start_char = answers["answer_start"][0]
            answer_text = answers["text"][0]
            answer_end_char = answer_start_char + len(answer_text)

            sequence_ids = tokenized.sequence_ids(i)

            context_start = None
            context_end = None
            for idx, sid in enumerate(sequence_ids):
                if sid == 1 and context_start is None:
                    context_start = idx
                if sid == 1:
                    context_end = idx

            if context_start is None or context_end is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            if not (offsets[context_start][0] <= answer_start_char and offsets[context_end][1] >= answer_end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                token_start = context_start
                while token_start <= context_end and offsets[token_start][0] <= answer_start_char:
                    token_start += 1
                start_pos = token_start - 1

                token_end = context_end
                while token_end >= context_start and offsets[token_end][1] >= answer_end_char:
                    token_end -= 1
                end_pos = token_end + 1

                start_positions.append(start_pos)
                end_positions.append(end_pos)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    tokenized_train = train_ds.map(
        prepare_features,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing TRAIN",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenized_train.save_to_disk(str(OUT_PATH))
    print("Saved tokenized train to:", OUT_PATH)
    print(tokenized_train)

if __name__ == "__main__":
    main()
