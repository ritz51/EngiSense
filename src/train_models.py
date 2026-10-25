"""Comprehensive training and evaluation pipeline for EngiSense.

Trains and evaluates:
1. XGBoost baseline (tabular features + rolling degradation statistics)
2. PyTorch LSTM (sliding sequence windows)
3. PyTorch GRU (sliding sequence windows)

Computes validation metrics and official NASA C-MAPSS test metrics (MAE, RMSE, R2, NASA Score).
Saves model checkpoints, scalers, feature configurations, evaluation plots, and comparison JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, Any, Tuple, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .data_loader import load_subset, PROJECT_ROOT
from .rul import compute_rul_train, compute_rul_test, clip_rul
from .metrics import compute_metrics, compute_nasa_score
from .feature_engineering import (
    get_feature_columns,
    split_engine_ids,
    extract_tabular_features,
    build_sequence_dataset,
    extract_last_sequence_per_engine,
)


# ---------------------------------------------------------------------------
# Sequence Neural Network Architectures (PyTorch)
# ---------------------------------------------------------------------------

class SequenceRULModel(nn.Module):
    """Deep Recurrent Model (LSTM or GRU) for Remaining Useful Life estimation."""

    def __init__(
        self,
        cell_type: str,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cell_type = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        rnn_cls = nn.LSTM if self.cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.ReLU(),  # Guarantees non-negative RUL prediction
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # last time step
        pred = self.head(last_hidden)
        return pred.squeeze(-1)


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def train_sequence_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 35,
    lr: float = 1e-3,
    patience: int = 7,
    checkpoint_path: pathlib.Path | None = None,
) -> Tuple[nn.Module, dict, List[float], List[float]]:
    """Train a PyTorch sequence model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_train_loss = 0.0
        n_train = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_train_loss += loss.item() * len(y_batch)
            n_train += len(y_batch)

        train_loss = running_train_loss / max(n_train, 1)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                running_val_loss += loss.item() * len(y_batch)
                n_val += len(y_batch)

        val_loss = running_val_loss / max(n_val, 1)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_name}] Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{model_name}] Epoch {epoch:02d}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    duration = time.time() - start_time

    # Load best weights
    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "training_seconds": round(duration, 2),
    }
    return model, info, train_losses, val_losses


def predict_sequence_model(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Run batched inference on a sequence array."""
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            preds.append(out.cpu().numpy())

    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# End-to-End Orchestrator
# ---------------------------------------------------------------------------

def train_and_evaluate_all(
    subset: str = "FD001",
    seq_len: int = 30,
    max_rul: int = 125,
    epochs: int = 35,
    batch_size: int = 64,
    hidden_size: int = 64,
    seed: int = 42,
    val_frac: float = 0.2,
    output_dir: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Train XGBoost, LSTM, and GRU models on the specified C-MAPSS subset.

    Evaluates models on held-out validation engines and official C-MAPSS test engines.
    """
    subset = subset.upper()
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "models" / subset
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" EngiSense: Training Models for {subset}")
    print(f" Seq Len: {seq_len} | Max RUL Cap: {max_rul} | Seed: {seed}")
    print(f"=======================================================")

    # 1. Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation Device: {device}")

    # 2. Load Data
    raw_train_df, raw_test_df, official_rul_series = load_subset(subset)
    print(f"Loaded {subset}: {len(raw_train_df)} train rows, {len(raw_test_df)} test rows, {len(official_rul_series)} test engines.")

    # 3. Target RUL calculation
    train_df = compute_rul_train(raw_train_df)
    if max_rul is not None and max_rul > 0:
        train_df = clip_rul(train_df, max_rul)

    # 4. Feature Selection
    feature_cols = get_feature_columns(train_df, subset=subset, drop_constants=True)
    print(f"Selected {len(feature_cols)} informative features: {feature_cols}")

    # 5. Split train/validation engines (leakage prevention)
    train_ids, val_ids = split_engine_ids(train_df, val_frac=val_frac, seed=seed)
    train_split = train_df[train_df["unit_id"].isin(train_ids)].copy()
    val_split = train_df[train_df["unit_id"].isin(val_ids)].copy()
    print(f"Engine Split: {len(train_ids)} train engines ({len(train_split)} rows), {len(val_ids)} val engines ({len(val_split)} rows)")

    # -----------------------------------------------------------------------
    # MODEL A: XGBoost Baseline
    # -----------------------------------------------------------------------
    print("\n--- [1/3] Training XGBoost Baseline ---")
    xgb_train_feat = extract_tabular_features(train_split, feature_cols)
    xgb_val_feat = extract_tabular_features(val_split, feature_cols)

    tabular_feature_cols = [c for c in xgb_train_feat.columns if c not in {"unit_id", "cycle", "RUL"}]

    xgb_scaler = StandardScaler()
    X_train_xgb = xgb_scaler.fit_transform(xgb_train_feat[tabular_feature_cols])
    y_train_xgb = xgb_train_feat["RUL"].to_numpy(dtype=np.float32)

    X_val_xgb = xgb_scaler.transform(xgb_val_feat[tabular_feature_cols])
    y_val_xgb = xgb_val_feat["RUL"].to_numpy(dtype=np.float32)

    xgb_start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        early_stopping_rounds=25,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train_xgb,
        y_train_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=False,
    )
    xgb_duration = time.time() - xgb_start

    # XGBoost Validation predictions
    xgb_val_preds = np.clip(xgb_model.predict(X_val_xgb), 0, None)
    xgb_val_metrics = compute_metrics(y_val_xgb, xgb_val_preds)

    # XGBoost Official Test predictions (last cycle row per test engine)
    test_enriched = extract_tabular_features(raw_test_df, feature_cols)
    test_last_rows = test_enriched.sort_values(["unit_id", "cycle"]).groupby("unit_id").last().reset_index()
    X_test_xgb = xgb_scaler.transform(test_last_rows[tabular_feature_cols])
    xgb_test_preds = np.clip(xgb_model.predict(X_test_xgb), 0, None)
    y_test_official = official_rul_series.to_numpy(dtype=np.float32)
    xgb_test_metrics = compute_metrics(y_test_official, xgb_test_preds)

    print(f"XGBoost Test Metrics: {xgb_test_metrics} (Trained in {xgb_duration:.2f}s)")

    # Save XGBoost Artifacts
    joblib.dump(xgb_model, output_dir / "best_xgboost.joblib")
    joblib.dump(xgb_scaler, output_dir / "xgb_scaler.joblib")

    # -----------------------------------------------------------------------
    # MODEL B & C: Sequence Datasets (LSTM & GRU)
    # -----------------------------------------------------------------------
    print("\n--- Preparing Sequence Datasets for LSTM & GRU ---")
    X_train_seq, y_train_seq, seq_scaler = build_sequence_dataset(
        train_split, feature_cols, seq_len=seq_len, fit_scaler=True
    )
    X_val_seq, y_val_seq, _ = build_sequence_dataset(
        val_split, feature_cols, seq_len=seq_len, scaler=seq_scaler, fit_scaler=False
    )
    joblib.dump(seq_scaler, output_dir / "scaler.joblib")

    # Official Test Sequences: last sequence window per test engine
    X_test_seq = extract_last_sequence_per_engine(
        raw_test_df, feature_cols, seq_len=seq_len, scaler=seq_scaler
    )

    print(f"Sequence Shapes: Train {X_train_seq.shape}, Val {X_val_seq.shape}, Test {X_test_seq.shape}")

    train_tensor_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
    val_tensor_dataset = TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq))

    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor_dataset, batch_size=batch_size, shuffle=False)

    n_features = len(feature_cols)

    # -----------------------------------------------------------------------
    # MODEL B: LSTM
    # -----------------------------------------------------------------------
    print("\n--- [2/3] Training PyTorch LSTM ---")
    lstm_model = SequenceRULModel(cell_type="LSTM", n_features=n_features, hidden_size=hidden_size, num_layers=2)
    lstm_checkpoint = output_dir / "best_lstm.pth"
    lstm_model, lstm_info, lstm_train_losses, lstm_val_losses = train_sequence_model(
        "LSTM", lstm_model, train_loader, val_loader, device,
        epochs=epochs, lr=1e-3, patience=7, checkpoint_path=lstm_checkpoint
    )

    lstm_val_preds = predict_sequence_model(lstm_model, X_val_seq, device)
    lstm_val_metrics = compute_metrics(y_val_seq, lstm_val_preds)

    lstm_test_preds = predict_sequence_model(lstm_model, X_test_seq, device)
    lstm_test_metrics = compute_metrics(y_test_official, lstm_test_preds)
    print(f"LSTM Test Metrics: {lstm_test_metrics}")

    # -----------------------------------------------------------------------
    # MODEL C: GRU
    # -----------------------------------------------------------------------
    print("\n--- [3/3] Training PyTorch GRU ---")
    gru_model = SequenceRULModel(cell_type="GRU", n_features=n_features, hidden_size=hidden_size, num_layers=2)
    gru_checkpoint = output_dir / "best_gru.pth"
    gru_model, gru_info, gru_train_losses, gru_val_losses = train_sequence_model(
        "GRU", gru_model, train_loader, val_loader, device,
        epochs=epochs, lr=1e-3, patience=7, checkpoint_path=gru_checkpoint
    )

    gru_val_preds = predict_sequence_model(gru_model, X_val_seq, device)
    gru_val_metrics = compute_metrics(y_val_seq, gru_val_preds)

    gru_test_preds = predict_sequence_model(gru_model, X_test_seq, device)
    gru_test_metrics = compute_metrics(y_test_official, gru_test_preds)
    print(f"GRU Test Metrics: {gru_test_metrics}")

    # -----------------------------------------------------------------------
    # Comparison & Model Selection
    # -----------------------------------------------------------------------
    models_summary = {
        "XGBoost": {
            "validation": xgb_val_metrics,
            "test": xgb_test_metrics,
            "training_seconds": round(xgb_duration, 2),
            "best_epoch": int(xgb_model.best_iteration) if hasattr(xgb_model, "best_iteration") else None,
            "architecture": "Gradient Boosted Decision Trees (Tabular + Rolling Stats)",
        },
        "LSTM": {
            "validation": lstm_val_metrics,
            "test": lstm_test_metrics,
            "training_seconds": lstm_info["training_seconds"],
            "best_epoch": lstm_info["best_epoch"],
            "architecture": "2-Layer PyTorch LSTM (Seq Window = 30)",
        },
        "GRU": {
            "validation": gru_val_metrics,
            "test": gru_test_metrics,
            "training_seconds": gru_info["training_seconds"],
            "best_epoch": gru_info["best_epoch"],
            "architecture": "2-Layer PyTorch GRU (Seq Window = 30)",
        },
    }

    # Best model selection by lowest test RMSE
    best_model_name = min(models_summary.keys(), key=lambda k: models_summary[k]["test"]["RMSE"])

    comparison_report = {
        "subset": subset,
        "recommended_model": best_model_name,
        "selection_criteria": "Lowest Test RMSE on official NASA test labels",
        "models": models_summary,
        "parameters": {
            "seq_len": seq_len,
            "max_rul_cap": max_rul,
            "val_frac": val_frac,
            "seed": seed,
            "features_used": feature_cols,
        },
    }

    comparison_path = output_dir / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")

    # Feature configuration for inference
    feature_config = {
        "subset": subset,
        "feature_cols": feature_cols,
        "tabular_feature_cols": tabular_feature_cols,
        "seq_len": seq_len,
        "max_rul": max_rul,
        "recommended_model": best_model_name,
    }
    (output_dir / "feature_config.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Comprehensive Visualizations
    # -----------------------------------------------------------------------
    try:
        # 1. Actual vs Predicted Comparison Plot
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
        test_preds_map = {
            "XGBoost": xgb_test_preds,
            "LSTM": lstm_test_preds,
            "GRU": gru_test_preds,
        }

        for idx, (m_name, p) in enumerate(test_preds_map.items()):
            ax = axes[idx]
            ax.scatter(y_test_official, p, alpha=0.7, color="#2563EB" if m_name == best_model_name else "#64748B", edgecolors="none", s=40)
            max_val = max(np.max(y_test_official), np.max(p)) + 10
            ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect Alignment (y = x)")
            rmse_val = models_summary[m_name]["test"]["RMSE"]
            mae_val = models_summary[m_name]["test"]["MAE"]
            nasa_val = models_summary[m_name]["test"]["NASA_Score"]
            ax.set_title(f"{m_name} (Test RMSE: {rmse_val:.2f}, MAE: {mae_val:.2f})", fontsize=12, fontweight="bold")
            ax.set_xlabel("True RUL (cycles)", fontsize=11)
            ax.set_ylabel("Predicted RUL (cycles)", fontsize=11)
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
        plt.close()

        # 2. Residual Distribution Plot
        plt.figure(figsize=(10, 5), dpi=150)
        for m_name, p in test_preds_map.items():
            residuals = p - y_test_official
            sns.kdeplot(residuals, label=f"{m_name} (mean: {residuals.mean():.1f}, std: {residuals.std():.1f})", linewidth=2)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1.2)
        plt.title(f"Prediction Error / Residual Distribution on Official Test Set ({subset})", fontsize=13, fontweight="bold")
        plt.xlabel("Residual (Predicted RUL - True RUL) [cycles]", fontsize=11)
        plt.ylabel("Density", fontsize=11)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "residual_analysis.png", dpi=150)
        plt.close()

        # 3. Model Comparison Bar Chart
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
        metrics_keys = ["RMSE", "MAE", "R2", "NASA_Score"]
        titles = ["RMSE (Lower is Better)", "MAE (Lower is Better)", "R² (Higher is Better)", "NASA Score (Lower is Better)"]
        colors = ["#3B82F6", "#10B981", "#8B5CF6"]

        for ax, m_key, title in zip(axes, metrics_keys, titles):
            vals = [models_summary[m]["test"][m_key] for m in ["XGBoost", "LSTM", "GRU"]]
            bars = ax.bar(["XGBoost", "LSTM", "GRU"], vals, color=colors, width=0.55)
            ax.set_title(title, fontsize=11, fontweight="bold")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=150)
        plt.close()

        # 4. Training Loss Curves (LSTM & GRU)
        plt.figure(figsize=(9, 4.5), dpi=150)
        plt.plot(range(1, len(lstm_train_losses) + 1), lstm_train_losses, label="LSTM Train Loss", color="#2563EB", linestyle="--")
        plt.plot(range(1, len(lstm_val_losses) + 1), lstm_val_losses, label="LSTM Val Loss", color="#2563EB", linewidth=2)
        plt.plot(range(1, len(gru_train_losses) + 1), gru_train_losses, label="GRU Train Loss", color="#10B981", linestyle="--")
        plt.plot(range(1, len(gru_val_losses) + 1), gru_val_losses, label="GRU Val Loss", color="#10B981", linewidth=2)
        plt.title(f"Sequence Models Training & Validation Loss ({subset})", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("MSE Loss", fontsize=11)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss.png", dpi=150)
        plt.close()

        # 5. XGBoost Feature Importance
        plt.figure(figsize=(10, 6), dpi=150)
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15
        plt.barh(range(len(indices)), importances[indices], color="#3B82F6", align="center")
        plt.yticks(range(len(indices)), [tabular_feature_cols[i] for i in indices])
        plt.title("Top 15 Feature Importances (XGBoost)", fontsize=13, fontweight="bold")
        plt.xlabel("Relative Importance", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance_xgboost.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"Warning: Visualization generation encountered an error: {e}")

    print("\n=======================================================")
    print(f" Evaluation Complete! Recommended Model: {best_model_name}")
    print(f" Comparison JSON saved to: outputs/models/{subset}/model_comparison.json")
    print(f"=======================================================\n")

    return comparison_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate all EngiSense models (XGBoost, LSTM, GRU)")
    parser.add_argument("--subset", type=str, default="FD001", help="C-MAPSS subset: FD001, FD002, FD003, FD004")
    parser.add_argument("--seq_len", type=int, default=30, help="Sliding sequence window length")
    parser.add_argument("--max_rul", type=int, default=125, help="Capped RUL value (default 125)")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs for neural networks")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden state size for LSTM and GRU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction of engines for validation")

    args = parser.parse_args()
    train_and_evaluate_all(
        subset=args.subset,
        seq_len=args.seq_len,
        max_rul=args.max_rul,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        seed=args.seed,
        val_frac=args.val_frac,
    )
