# src/finetuning/t5_finetune.py
"""
T5 (trainable) → Fine-tuned Embeddings → Random Forest → Score
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from torch.utils.data        import DataLoader
from torch.optim             import AdamW
from transformers            import T5Tokenizer, T5EncoderModel
from transformers            import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestRegressor

from src.finetuning.dataset        import T5ASAGDataset
from src.finetuning.sbert_finetune import (get_device,
                                           _compute_metrics,
                                           _print_metrics)


class T5Encoder(nn.Module):
    """
    T5-small encoder + simple regression head.
    Uses mean pooling over all tokens.
    After training → discard head → extract embeddings → RF.
    """

    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.encoder     = T5EncoderModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.d_model   # 512

        self.finetune_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def _mean_pool(self, hidden_state, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled

    def get_embeddings(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask)
        return self._mean_pool(output.last_hidden_state, attention_mask)

    def forward(self, input_ids, attention_mask):
        emb = self.get_embeddings(input_ids, attention_mask)
        return self.finetune_head(emb).squeeze(-1)


def train_t5(
    dataset_path,
    save_path,
    predictions_path,
    epochs     = 5,
    batch_size = 16,
    lr         = 1e-4,
    max_length = 128
):
    device = get_device()
    print(f"  [T5 → RF] Using: {device}")

    # ── Load Data ─────────────────────────────
    df        = pd.read_csv(dataset_path)
    score_min = df["score"].min()
    score_max = df["score"].max()
    df["score_normalized"] = (
        (df["score"] - score_min) / (score_max - score_min + 1e-9)
    )

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df  = train_test_split(train_df, test_size=0.1, random_state=42)
    test_df_reset     = test_df.reset_index(drop=True)

    print(f"  Train:{len(train_df)} | Val:{len(val_df)} | Test:{len(test_df)}")

    # ── Tokenizer ─────────────────────────────
    model_name = "t5-small"
    tokenizer  = T5Tokenizer.from_pretrained(model_name)

    train_loader = DataLoader(
        T5ASAGDataset(train_df, tokenizer, max_length),
        batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2
    )
    val_loader = DataLoader(
        T5ASAGDataset(val_df, tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )
    train_embed_loader = DataLoader(
        T5ASAGDataset(train_df, tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )
    test_loader = DataLoader(
        T5ASAGDataset(test_df_reset, tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )

    # ── PHASE 1: Fine-tune T5 ─────────────────
    print("\n  ── PHASE 1: Fine-tuning T5 ──")

    model     = T5Encoder(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = total_steps // 10,
        num_training_steps= total_steps
    )
    criterion     = nn.MSELoss()
    scaler        = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tgt  = batch["score"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                preds = model(ids, mask)
                loss  = criterion(preds, tgt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tgt  = batch["score"].to(device)
                val_loss += criterion(model(ids, mask), tgt).item()

        avg_tr  = train_loss / len(train_loader)
        avg_val = val_loss   / len(val_loader)
        print(f"    Epoch {epoch+1}/{epochs} | "
              f"train={avg_tr:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print("      ✅ best model saved")

    # ── PHASE 2: Extract Embeddings ───────────
    print("\n  ── PHASE 2: Extracting fine-tuned embeddings ──")

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    model.eval()

    def extract_embeddings(loader):
        all_emb, all_scores = [], []
        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                emb  = model.get_embeddings(ids, mask)
                all_emb.append(emb.cpu().float().numpy())
                all_scores.append(batch["score"].numpy())
        return np.vstack(all_emb), np.concatenate(all_scores)

    X_train, y_train = extract_embeddings(train_embed_loader)
    X_test,  y_test  = extract_embeddings(test_loader)

    print(f"    Train embeddings: {X_train.shape}")
    print(f"    Test  embeddings: {X_test.shape}")

    y_train_orig = y_train * (score_max - score_min) + score_min
    y_test_orig  = y_test  * (score_max - score_min) + score_min

    # ── PHASE 3: Random Forest ────────────────
    print("\n  ── PHASE 3: Training Random Forest ──")

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train_orig)

    rf_path = save_path.replace(".pt", "_rf.joblib")
    joblib.dump(rf, rf_path)
    print(f"    💾 RF saved → {rf_path}")

    # ── PHASE 4: Predict & Save ───────────────
    print("\n  ── PHASE 4: Predicting & Saving ──")

    all_preds  = rf.predict(X_test)
    all_labels = y_test_orig

    pred_df = pd.DataFrame({
        "reference_answer": test_df_reset["reference_answer"].values,
        "student_answer":   test_df_reset["student_answer"].values,
        "score":            test_df_reset["score"].values,
        "predicted_score":  np.round(all_preds, 3)
    })

    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    pred_df.to_csv(predictions_path, index=False)
    print(f"    📄 Predictions → {predictions_path}")

    metrics = _compute_metrics(all_labels, all_preds)
    _print_metrics(metrics)
    return metrics