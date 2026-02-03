from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

# -------------------------
# CONFIG
# -------------------------
MODEL_CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

BASE_DIR = Path("/kaggle/working/Projet") 
TRAIN_PATH = BASE_DIR / "Data" / "processed" / "train_squad_tokenized" / MODEL_CHECKPOINT
DEV_PATH   = BASE_DIR / "Data" / "processed" / "dev_squad_tokenized" / MODEL_CHECKPOINT


OUT_DIR = BASE_DIR / "models" / "bert_squad_finetuned_bert"

print("BASE_DIR =", BASE_DIR)
print("TRAIN_PATH =", TRAIN_PATH)
print("DEV_PATH   =", DEV_PATH)
print("OUT_DIR    =", OUT_DIR)

assert TRAIN_PATH.exists(), f"TRAIN_PATH not found: {TRAIN_PATH}"
assert DEV_PATH.exists(), f"DEV_PATH not found: {DEV_PATH}"

# Load tokenized datasets
train_ds = load_from_disk(str(TRAIN_PATH))
dev_ds   = load_from_disk(str(DEV_PATH))

# Load BERT model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

OUT_DIR.mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    logging_steps=200,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=DefaultDataCollator(),
    tokenizer=tokenizer,
)

print("🚀 TRAINING START")
trainer.train()

print("💾 SAVING MODEL TO:", OUT_DIR)
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print("✅ DONE. Files saved:")
import os
print(os.listdir(OUT_DIR))
