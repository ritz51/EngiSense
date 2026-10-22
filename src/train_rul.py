# EngineLife - train_rul.py
"""Reproducible training pipeline for LSTM and GRU models on NASA C-MAPSS."""

import argparse
import json
import pathlib
import time
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from .data_loader import load_subset
from .rul import compute_rul_train, compute_rul_test


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _engine_column(df: pd.DataFrame) -> str:
    """Return the column identifying each engine."""
    if "unit_id" in df.columns:
        return "unit_id"
    if "unit" in df.columns:
        return "unit"
    raise ValueError(
        "Engine ID column not found. Expected 'unit_id' or 'unit'."
    )


def _sensor_columns(df: pd.DataFrame) -> List[str]:
    """Return all sensor feature columns."""
    return [c for c in df.columns if c.startswith("sensor_")]


def _split_engine_ids(
    df: pd.DataFrame,
    val_frac: float,
    rng: np.random.Generator
) -> Tuple[List[int], List[int]]:
    """Split engines into training and validation groups."""

    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")

    engine_col = _engine_column(df)

    # FIX: Create a writable copy before shuffling.
    engine_ids = df[engine_col].drop_duplicates().to_numpy().copy()

    rng.shuffle(engine_ids)

    split = int(len(engine_ids) * (1 - val_frac))

    if split == 0 or split == len(engine_ids):
        raise ValueError(
            "Not enough engines for the requested training/validation split."
        )

    return (
        engine_ids[:split].tolist(),
        engine_ids[split:].tolist()
    )


# ---------------------------------------------------------------------
# Dataset and sequence generation
# ---------------------------------------------------------------------

class EngineDataset(Dataset):
    """Create fixed-length sensor sequences for RUL prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        scaler: StandardScaler | None = None
    ) -> None:

        if seq_len < 1:
            raise ValueError("seq_len must be at least 1.")

        self.seq_len = seq_len
        self.engine_col = _engine_column(df)
        self.feature_cols = _sensor_columns(df)

        if not self.feature_cols:
            raise ValueError("No sensor columns found in the dataset.")

        df = df.copy()

        if "RUL" not in df.columns:
            df = compute_rul_train(df)

        raw_seqs: List[np.ndarray] = []
        targets: List[float] = []

        for _, group in df.groupby(self.engine_col):
            group = group.sort_values("cycle")

            sensor_values = group[self.feature_cols].to_numpy(
                dtype=np.float32
            )
            rul_values = group["RUL"].to_numpy(dtype=np.float32)

            for start in range(len(group) - seq_len + 1):
                end = start + seq_len

                raw_seqs.append(sensor_values[start:end])
                targets.append(float(rul_values[end - 1]))

        if not raw_seqs:
            raise ValueError(
                f"No sequences generated. Check seq_len={seq_len} "
                "and the number of cycles per engine."
            )

        self.raw_seqs = np.stack(raw_seqs)
        self.targets = np.asarray(targets, dtype=np.float32)

        flattened = self.raw_seqs.reshape(
            -1, self.raw_seqs.shape[-1]
        )

        # Fit the scaler only when one is not supplied.
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(flattened)
        else:
            self.scaler = scaler

        scaled = self.scaler.transform(flattened)

        self.seqs = scaled.reshape(self.raw_seqs.shape).astype(
            np.float32
        )

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        sequence = torch.from_numpy(self.seqs[idx])
        target = torch.tensor(
            [self.targets[idx]], dtype=torch.float32
        )

        return sequence, target


# ---------------------------------------------------------------------
# LSTM and GRU models
# ---------------------------------------------------------------------

class BaseRULModel(nn.Module):

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: int = 1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = None
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.size(0)

        h0 = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_size,
            device=x.device
        )

        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(
                self.n_layers,
                batch_size,
                self.hidden_size,
                device=x.device
            )

            _, (hn, _) = self.rnn(x, (h0, c0))
        else:
            _, hn = self.rnn(x, h0)

        last_hidden = hn[-1]

        return self.fc(last_hidden)


class LSTMModel(BaseRULModel):

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: int = 1
    ):
        super().__init__(n_features, hidden_size, n_layers)

        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )


class GRUModel(BaseRULModel):

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: int = 1
    ):
        super().__init__(n_features, hidden_size, n_layers)

        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )


# ---------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device
) -> float:

    model.train()
    total_loss = 0.0

    for seq, target in loader:
        seq = seq.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(seq)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seq.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:

    model.eval()

    total_loss = 0.0
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for seq, target in loader:
            seq = seq.to(device)
            target = target.to(device)

            output = model(seq)
            loss = criterion(output, target)

            total_loss += loss.item() * seq.size(0)

            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    pred_array = np.concatenate(predictions).ravel()
    target_array = np.concatenate(targets).ravel()

    return avg_loss, pred_array, target_array


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:

    mae = float(np.mean(np.abs(y_true - y_pred)))

    rmse = float(
        np.sqrt(np.mean((y_true - y_pred) ** 2))
    )

    if np.allclose(y_true, y_true.mean()):
        r2 = None
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------

def run_training(
    subset: str,
    seq_len: int = 30,
    batch_size: int = 64,
    epochs: int = 30,
    hidden_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    seed: int = 42,
    patience: int = 5,
) -> None:

    subset = subset.upper()

    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(
            "Invalid subset. Choose FD001, FD002, FD003 or FD004."
        )

    if epochs < 1 or batch_size < 1 or hidden_size < 1:
        raise ValueError(
            "epochs, batch_size and hidden_size must be positive."
        )

    if lr <= 0 or patience < 1:
        raise ValueError(
            "Learning rate must be positive and patience at least 1."
        )

    print(
        f"\n=== Training RUL models for {subset} "
        f"(seq_len={seq_len}) ===",
        flush=True
    )

    # Set random seeds for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load data.
    train_df, test_df, rul_ref = load_subset(subset)

    train_df = compute_rul_train(train_df)

    print(f"Training rows: {len(train_df)}", flush=True)
    print(f"Test rows: {len(test_df)}", flush=True)

    # 2. Split by engine to avoid leakage.
    rng = np.random.default_rng(seed)

    train_ids, val_ids = _split_engine_ids(
        train_df, val_frac, rng
    )

    eng_col = _engine_column(train_df)

    train_split = train_df[
        train_df[eng_col].isin(train_ids)
    ].copy()

    val_split = train_df[
        train_df[eng_col].isin(val_ids)
    ].copy()

    print(
        f"Training engines: {len(train_ids)}; "
        f"validation engines: {len(val_ids)}",
        flush=True
    )

    # 3. Build sequences and scale sensor features.
    train_dataset = EngineDataset(
        train_split, seq_len=seq_len
    )

    val_dataset = EngineDataset(
        val_split,
        seq_len=seq_len,
        scaler=train_dataset.scaler
    )

    # Attach official test RUL values to test sequences.
    test_df = compute_rul_test(test_df, rul_ref)

    test_dataset = EngineDataset(
        test_df,
        seq_len=seq_len,
        scaler=train_dataset.scaler
    )

    print(f"Training sequences: {len(train_dataset)}", flush=True)
    print(f"Validation sequences: {len(val_dataset)}", flush=True)
    print(f"Test sequences: {len(test_dataset)}", flush=True)

    # 4. Create data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 5. Configure models.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}", flush=True)

    n_features = len(train_dataset.feature_cols)

    models = {
        "LSTM": LSTMModel(n_features, hidden_size),
        "GRU": GRUModel(n_features, hidden_size)
    }

    criterion = nn.MSELoss()

    # 6. Create output directory.
    project_root = pathlib.Path(__file__).resolve().parents[1]

    out_dir = (
        project_root / "outputs" / "models" / subset
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        train_dataset.scaler,
        out_dir / "scaler.joblib"
    )

    history_path = out_dir / "training_history.csv"

    with open(history_path, "w", encoding="utf-8") as file:
        file.write("model,epoch,train_loss,val_loss\n")

    # 7. Train each model.
    for name, model in models.items():

        print(f"\n--- Training {name} ---", flush=True)

        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        best_val = float("inf")
        best_epoch = -1
        epochs_no_improve = 0

        start_time = time.time()

        for epoch in range(1, epochs + 1):

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            val_loss, _, _ = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            with open(
                history_path, "a", encoding="utf-8"
            ) as file:
                file.write(
                    f"{name},{epoch},{train_loss:.6f},"
                    f"{val_loss:.6f}\n"
                )

            print(
                f"Epoch {epoch:02d}/{epochs}: "
                f"train loss={train_loss:.4f}, "
                f"val loss={val_loss:.4f}",
                flush=True
            )

            # Save the best model checkpoint.
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                epochs_no_improve = 0

                checkpoint = (
                    out_dir / f"best_{name.lower()}.pth"
                )

                torch.save(model.state_dict(), checkpoint)

            else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping {name} at epoch "
                        f"{epoch}.",
                        flush=True
                    )
                    break

        duration = time.time() - start_time

        # Load best checkpoint and evaluate.
        best_checkpoint = (
            out_dir / f"best_{name.lower()}.pth"
        )

        model.load_state_dict(
            torch.load(
                best_checkpoint,
                map_location=device,
                weights_only=True
            )
        )

        _, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        _, test_preds, test_targets = evaluate(
            model, test_loader, criterion, device
        )

        val_metrics = compute_metrics(
            val_targets, val_preds
        )

        test_metrics = compute_metrics(
            test_targets, test_preds
        )

        metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "best_epoch": best_epoch,
            "training_seconds": duration
        }

        metrics_path = (
            out_dir / f"metrics_{name.lower()}.json"
        )

        metrics_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=False),
            encoding="utf-8"
        )

        print(
            f"{name} validation metrics: {val_metrics}",
            flush=True
        )

        print(
            f"{name} test metrics: {test_metrics}",
            flush=True
        )

        # Plot loss curves and validation predictions.
        try:
            import matplotlib.pyplot as plt

            history_df = pd.read_csv(history_path)
            model_history = history_df[
                history_df["model"] == name
            ]

            plt.figure(figsize=(8, 5))
            plt.plot(
                model_history["epoch"],
                model_history["train_loss"],
                label="Training loss"
            )
            plt.plot(
                model_history["epoch"],
                model_history["val_loss"],
                label="Validation loss"
            )
            plt.title(f"{name} Loss - {subset}")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                out_dir / f"loss_{name.lower()}.png",
                dpi=150
            )
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.scatter(
                val_targets,
                val_preds,
                alpha=0.6
            )

            max_value = max(
                float(np.max(val_targets)),
                float(np.max(val_preds))
            )

            plt.plot(
                [0, max_value],
                [0, max_value],
                "--"
            )

            plt.xlabel("True RUL")
            plt.ylabel("Predicted RUL")
            plt.title(f"{name} Validation Predictions")
            plt.tight_layout()
            plt.savefig(
                out_dir / f"val_scatter_{name.lower()}.png",
                dpi=150
            )
            plt.close()

        except Exception as error:
            print(
                f"Plot generation failed for {name}: {error}",
                flush=True
            )

    print(
        "\nTraining complete. Artifacts saved under:",
        out_dir,
        flush=True
    )


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train LSTM and GRU models for RUL prediction"
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        help="Dataset subset: FD001, FD002, FD003 or FD004"
    )

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    run_training(
        subset=args.subset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        lr=args.lr,
        seed=args.seed,
        val_frac=args.val_frac,
        patience=args.patience,
    )