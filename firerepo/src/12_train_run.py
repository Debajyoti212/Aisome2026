"""
AISoMe 2026 Squad4 - Classifier Training Script (src/12_train_run.py)

Trains fine-tuned multilingual Transformer stance classifiers (MuRIL / XLM-RoBERTa)
on translated Hindi and Bengali social-media comments.

Supported runs in paper:
  Model 1 (run1_muril_leaky):
    python src/12_train_run.py --model google/muril-base-cased --tag run1_muril_leaky --seed 42 --epochs 4 --lr 2e-5
  Model 2 (run2_muril_group):
    python src/12_train_run.py --model google/muril-base-cased --tag run2_muril_group --seed 7 --epochs 4 --lr 2e-5 --group_split
  Model 3 (run3_xlmr_group):
    python src/12_train_run.py --model xlm-roberta-base --tag run3_xlmr_group --seed 7 --epochs 5 --lr 1.5e-5 --group_split
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LABEL2ID = {"Favour": 0, "Against": 1, "None": 2}
ID2LABEL = {0: "Favour", 1: "Against", 2: "None"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StanceDataset(Dataset):
    def __init__(self, texts, labels, languages, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        lang = self.languages[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["lang"] = lang
        return item


def load_dataset(data_dir: Path):
    """
    Loads translated Hindi and Bengali stance training datasets.
    Creates mock sample data if files are not present.
    """
    hi_path = data_dir / "train_data_hi.csv"
    bn_path = data_dir / "train_data_bn.csv"
    pooled_path = data_dir / "translated_train.csv"

    if pooled_path.exists():
        df = pd.read_csv(pooled_path)
    elif hi_path.exists() and bn_path.exists():
        df_hi = pd.read_csv(hi_path)
        df_bn = pd.read_csv(bn_path)
        df_hi["lang"] = "hi"
        df_bn["lang"] = "bn"
        df = pd.concat([df_hi, df_bn], ignore_index=True)
    else:
        logger.warning(f"Dataset not found at {data_dir}. Generating synthetic placeholder data for demonstration.")
        records = []
        labels = ["Favour", "Against", "None"]
        for i in range(500):
            source_id = f"src_{i}"
            lbl = labels[i % 3]
            records.append({
                "source_id": source_id,
                "text": f"जलवायु परिवर्तन चिंता का विषय है {i}",
                "label": lbl,
                "lang": "hi"
            })
            records.append({
                "source_id": source_id,
                "text": f"জলবায়ু পরিবর্তন উদ্বেগের বিষয় {i}",
                "label": lbl,
                "lang": "bn"
            })
        df = pd.DataFrame(records)

    if "source_id" not in df.columns:
        df["source_id"] = [f"src_{i // 2}" for i in range(len(df))]

    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def split_dataset(df: pd.DataFrame, group_split: bool, seed: int, split_ratio: float = 0.90):
    test_size = 1.0 - split_ratio

    if group_split:
        logger.info("Using source-grouped stratified 90/10 split (prevents cross-lingual data leakage).")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, val_idx = next(gss.split(df, groups=df["source_id"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
    else:
        logger.info("Using instance-level stratified 90/10 split.")
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["label_id"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def compute_metrics(eval_preds, eval_langs):
    preds, labels = eval_preds
    overall_macro_f1 = f1_score(labels, preds, average="macro")
    overall_acc = accuracy_score(labels, preds)

    # Sub-metrics per language
    langs = np.array(eval_langs)
    hi_mask = (langs == "hi")
    bn_mask = (langs == "bn")

    hi_f1 = f1_score(labels[hi_mask], preds[hi_mask], average="macro") if np.sum(hi_mask) > 0 else 0.0
    bn_f1 = f1_score(labels[bn_mask], preds[bn_mask], average="macro") if np.sum(bn_mask) > 0 else 0.0

    return {
        "overall_macro_f1": float(overall_macro_f1),
        "bengali_macro_f1": float(bn_f1),
        "hindi_macro_f1": float(hi_f1),
        "overall_accuracy": float(overall_acc),
    }


def train_model(args):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {data_dir}...")
    df = load_dataset(data_dir)

    train_df, val_df = split_dataset(df, args.group_split, args.seed, args.split_ratio)
    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Tokenizer & Model setup
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Balanced class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"].values,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    train_dataset = StanceDataset(
        train_df["text"].values, train_df["label_id"].values, train_df["lang"].values, tokenizer, args.max_length
    )
    val_dataset = StanceDataset(
        val_df["text"].values, val_df["label_id"].values, val_df["lang"].values, tokenizer, args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and torch.cuda.is_available()))

    best_macro_f1 = 0.0
    best_metrics = {}

    logger.info(f"Starting training for {args.epochs} epochs using {args.model}...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(args.precision == "fp16" and torch.cuda.is_available())):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        val_preds = []
        val_labels = []
        val_langs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=(args.precision == "fp16" and torch.cuda.is_available())):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                val_langs.extend(batch["lang"])

        metrics = compute_metrics((np.array(val_preds), np.array(val_labels)), val_langs)
        logger.info(
            f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} | "
            f"Macro-F1: {metrics['overall_macro_f1']:.4f} (BN: {metrics['bengali_macro_f1']:.4f}, HI: {metrics['hindi_macro_f1']:.4f}) | "
            f"Accuracy: {metrics['overall_accuracy']:.4f}"
        )

        if metrics["overall_macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["overall_macro_f1"]
            best_metrics = metrics
            logger.info(f"Saving best model checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            with open(output_dir / "eval_results.json", "w") as f:
                json.dump(best_metrics, f, indent=2)

    logger.info("Training complete.")
    logger.info(f"Best Validation Results for {args.tag}: {best_metrics}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stance Detection Classifier for AISoMe 2026 Squad4")
    parser.add_argument("--model", type=str, default="google/muril-base-cased", help="Pretrained model identifier")
    parser.add_argument("--tag", type=str, default="run1_muril_leaky", help="Run identifier / output tag")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--group_split", action="store_true", help="Use source-grouped stratified split")
    parser.add_argument("--split_ratio", type=float, default=0.90, help="Train/Validation split ratio")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size per device")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="Floating point precision")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing translated training CSVs")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for saved model artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
