# src/finetuning/sbert_finetune.py
"""
SBERT (trainable) → Fine-tuned Embeddings → Random Forest → Score
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from torch.utils.data        import DataLoader
from torch.optim             import AdamW
from transformers            import AutoTokenizer, AutoModel
from transformers            import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import (mean_squared_error,
                                     mean_absolute_error,
                                     cohen_kappa_score)
from scipy.stats             import pearsonr

from src.finetuning.dataset import ASAGDataset


# ══════════════════════════════════════════════
#  GPU Selection — explicitly use RTX 4070
# ══════════════════════════════════════════════
def get_device():
    """
    Explicitly select RTX 4070 (discrete GPU).
    In laptops with 2 GPUs:
        cuda:0 = Intel/iGPU  (sometimes)
        cuda:1 = RTX 4070    (discrete)
    We pick the one with most free memory.
    """
    if not torch.cuda.is_available():
        print("  ⚠️  No CUDA found. Using CPU.")
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()
    print(f"  Found {num_gpus} GPU(s):")

    best_gpu  = 0
    best_free = 0

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        # Get free memory
        torch.cuda.set_device(i)
        free  = (props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3
        print(f"    GPU {i}: {props.name} | Total={total:.1f}GB | Free={free:.1f}GB")

        if "4070" in props.name or free > best_free:
            best_free = free
            best_gpu  = i

    print(f"  ✅ Selected GPU {best_gpu}: "
          f"{torch.cuda.get_device_properties(best_gpu).name}")

    return torch.device(f"cuda:{best_gpu}")


# ══════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════
class SBERTEncoder(nn.Module):
    """
    SBERT backbone + simple regression head for fine-tuning.
    After training → discard head → extract embeddings → RF.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder     = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size   # 384

        self.finetune_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def mean_pooling(self, model_output, attention_mask):
        token_emb     = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (
            torch.sum(token_emb * mask_expanded, dim=1)
            / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        )

    def get_embeddings(self, input_ids, attention_mask):
        """Encoder only — no head. Used for RF input."""
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask)
        return self.mean_pooling(output, attention_mask)     # (B, 384)

    def forward(self, input_ids, attention_mask):
        """Encoder + head. Used during fine-tuning."""
        emb = self.get_embeddings(input_ids, attention_mask)
        return self.finetune_head(emb).squeeze(-1)           # (B,)


# ══════════════════════════════════════════════
#  Training Pipeline
# ══════════════════════════════════════════════
def train_sbert(
    dataset_path,
    save_path,
    predictions_path,
    epochs     = 5,
    batch_size = 16,
    lr         = 2e-5,
    max_length = 128
):
    device = get_device()
    print(f"  [SBERT → RF] Using: {device}")

    # ── Load & Normalize Data ─────────────────
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

    # ── Tokenizer & DataLoaders ───────────────
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    train_loader = DataLoader(
        ASAGDataset(train_df, tokenizer, max_length),
        batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2
    )
    val_loader = DataLoader(
        ASAGDataset(val_df, tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )
    # No shuffle for embedding extraction
    train_embed_loader = DataLoader(
        ASAGDataset(train_df, tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )
    test_loader = DataLoader(
        ASAGDataset(test_df_reset, tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2
    )

    # ══════════════════════════════════════════
    #  PHASE 1 — Fine-tune SBERT
    # ══════════════════════════════════════════
    print("\n  ── PHASE 1: Fine-tuning SBERT ──")

    model     = SBERTEncoder(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = total_steps // 10,
        num_training_steps= total_steps
    )
    criterion = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):

        # Train
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

        # Validate
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

    # ══════════════════════════════════════════
    #  PHASE 2 — Extract Fine-tuned Embeddings
    # ══════════════════════════════════════════
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

    # De-normalize to original scale for RF
    y_train_orig = y_train * (score_max - score_min) + score_min
    y_test_orig  = y_test  * (score_max - score_min) + score_min

    # ══════════════════════════════════════════
    #  PHASE 3 — Train Random Forest
    # ══════════════════════════════════════════
    print("\n  ── PHASE 3: Training Random Forest on embeddings ──")

    rf = RandomForestRegressor(
        n_estimators      = 300,
        max_depth         = 20,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        random_state      = 42,
        n_jobs            = -1
    )
    rf.fit(X_train, y_train_orig)

    rf_path = save_path.replace(".pt", "_rf.joblib")
    joblib.dump(rf, rf_path)
    print(f"    💾 RF saved → {rf_path}")

    # ══════════════════════════════════════════
    #  PHASE 4 — Predict & Save
    # ══════════════════════════════════════════
    print("\n  ── PHASE 4: Predicting & Saving ──")

    all_preds  = rf.predict(X_test)
    all_labels = y_test_orig

    # Save predictions CSV (matching your screenshot format)
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


# ══════════════════════════════════════════════
#  Shared Helpers
# ══════════════════════════════════════════════
def _compute_metrics(labels, preds):
    pearson, _ = pearsonr(labels, preds)
    rmse       = np.sqrt(mean_squared_error(labels, preds))
    mae        = mean_absolute_error(labels, preds)

    int_labels = np.round(labels).astype(int)
    int_preds  = np.round(preds).astype(int)
    int_preds  = np.clip(int_preds, int_labels.min(), int_labels.max())

    try:
        qwk = cohen_kappa_score(int_labels, int_preds, weights="quadratic")
    except Exception:
        qwk = 0.0

    return {"pearson": pearson, "rmse": rmse, "mae": mae, "qwk": qwk}


def _print_metrics(m):
    print(f"    Pearson={m['pearson']:.4f}  RMSE={m['rmse']:.4f}  "
          f"MAE={m['mae']:.4f}  QWK={m['qwk']:.4f}")