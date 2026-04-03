"""
train_baseline_finbert.py – Train & evaluate the Hierarchical **FinBERT** baseline with
logging of overall **MACRO F1**, **Balanced Accuracy**, **Recall** and **Precision**.

Key changes vs. previous version
--------------------------------
1. Added `balanced_accuracy_score` from `sklearn.metrics`.
2. Metrics now computed with `average="macro"` for F1, Recall, Precision.
3. Prints & CSV logging updated to include the new metrics.
4. Function signatures adjusted accordingly.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import HierarchicalDataset
from model import HierarchicalBert

# ─── Configuration ────────────────────────────────────────────────────────────
EPOCHS = 3
BATCH_SIZE = 2  # Keep batch size small due to model size
LEARNING_RATE = 2e-5
MODEL_NAME = "ProsusAI/finbert"  # ← FinBERT checkpoint on Hugging Face Hub
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 1
N_SPLITS = 5  # Number of folds for cross-validation
DATA_FILE = "merged_data_nasdaq.csv"
RESULTS_FILE = "baselines/results.csv"

SEGMENT_LENGTHS_L = [32, 64, 128]


def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    """Performs one full training pass over the data."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        segment_lengths = batch["segment_lengths"]  # kept on CPU

        # Forward + backward
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, segment_lengths)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, loss_fn):
    """Evaluates the model and returns loss + metrics (macro‑averaged)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            segment_lengths = batch["segment_lengths"]

            outputs = model(input_ids, attention_mask, segment_lengths)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Standard (micro) accuracy kept for reference
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, f1_macro, bal_acc, prec_macro, rec_macro, acc


def main():
    """Main function to orchestrate training and evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. Load & clean data
    print("Loading data…")
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["transcript", "future_3bday_cum_return"]).reset_index(drop=True)

    overall_best_f1 = -1.0
    overall_best_l = None
    results_log = []

    # 2. Hyper‑parameter search over segment length L
    for l_val in SEGMENT_LENGTHS_L:
        print(f"\n{'='*20} Testing Segment Length (L) = {l_val} {'='*20}")

        # Create dataset for this L
        full_dataset = HierarchicalDataset(df, tokenizer, max_segment_len=l_val)

        # 5‑fold CV setup
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
            print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

            # Fresh model each fold – with FinBERT backbone
            model = HierarchicalBert(
                model_name=MODEL_NAME,
                lstm_hidden_size=LSTM_HIDDEN_SIZE,
                lstm_layers=LSTM_LAYERS,
            ).to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )

            # Training
            all_val_preds, all_val_labels = [], []
            for epoch in range(EPOCHS):
                print(f"  -- Epoch {epoch + 1}/{EPOCHS} --")
                tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
                print(f"  Avg training loss: {tr_loss:.4f}")

                val_loss, f1_macro, bal_acc, prec_macro, rec_macro, acc = evaluate(
                    model, val_loader, device, loss_fn
                )
                print(
                    f"  Val loss: {val_loss:.4f} | Macro F1: {f1_macro:.4f} | "
                    f"Balanced Acc: {bal_acc:.4f} | Macro Precision: {prec_macro:.4f} | "
                    f"Macro Recall: {rec_macro:.4f} | Acc: {acc:.4f}"
                )

                # Store predictions for fold‑level metrics
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    seg_len = batch["segment_lengths"]
                    outputs = model(input_ids, attention_mask, seg_len)
                    preds = torch.argmax(outputs, dim=1)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

            # Fold‑level metrics
            fold_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro", zero_division=0)
            fold_bal_acc = balanced_accuracy_score(all_val_labels, all_val_preds)
            fold_prec_macro = precision_score(all_val_labels, all_val_preds, average="macro", zero_division=0)
            fold_rec_macro = recall_score(all_val_labels, all_val_preds, average="macro", zero_division=0)

            print(
                f"  Overall Macro F1 for Fold {fold + 1}: {fold_f1_macro:.4f} | "
                f"Balanced Acc: {fold_bal_acc:.4f} | Macro Precision: {fold_prec_macro:.4f} | "
                f"Macro Recall: {fold_rec_macro:.4f}"
            )

            fold_results.append(
                {
                    "macro_f1": fold_f1_macro,
                    "balanced_accuracy": fold_bal_acc,
                    "macro_precision": fold_prec_macro,
                    "macro_recall": fold_rec_macro,
                }
            )

        # Average metrics for this L
        avg_f1 = np.mean([r["macro_f1"] for r in fold_results])
        avg_bal_acc = np.mean([r["balanced_accuracy"] for r in fold_results])
        avg_prec = np.mean([r["macro_precision"] for r in fold_results])
        avg_rec = np.mean([r["macro_recall"] for r in fold_results])

        print(f"\n--- Average Metrics for L = {l_val} ---")
        print(
            f"Avg Macro F1: {avg_f1:.4f} | Avg Balanced Acc: {avg_bal_acc:.4f} | "
            f"Avg Macro Precision: {avg_prec:.4f} | Avg Macro Recall: {avg_rec:.4f}"
        )

        results_log.append(
            {
                "segment_length_L": l_val,
                "avg_macro_f1": avg_f1,
                "avg_balanced_accuracy": avg_bal_acc,
                "avg_macro_precision": avg_prec,
                "avg_macro_recall": avg_rec,
            }
        )

        # Track best L
        if avg_f1 > overall_best_f1:
            overall_best_f1 = avg_f1
            overall_best_l = l_val

    print(f"\n{'='*20} Cross‑Validation Complete {'='*20}")
    print(
        f"Best segment length (L) by Macro F1: {overall_best_l} "
        f"(Macro F1 = {overall_best_f1:.4f})"
    )

    # Persist results
    pd.DataFrame(results_log).to_csv(RESULTS_FILE, index=False)
    print(f"Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
