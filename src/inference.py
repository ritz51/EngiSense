"""Inference engine for EngiSense.

Loads saved model artifacts (XGBoost, LSTM, GRU) and feature scalers.
Validates input engine sensor observations, executes real model inference,
and computes maintenance guidance and health risk classifications.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from .data_loader import load_subset, PROJECT_ROOT, assign_column_names
from .feature_engineering import (
    extract_tabular_features,
    pad_or_truncate_sequence,
)
from .train_models import SequenceRULModel


class InferenceEngine:
    """Manages loaded artifacts and executes real-time RUL prediction."""

    def __init__(self, subset: str = "FD001", model_dir: pathlib.Path | None = None):
        self.subset = subset.upper()
        if model_dir is None:
            self.model_dir = PROJECT_ROOT / "outputs" / "models" / self.subset
        else:
            self.model_dir = pathlib.Path(model_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_config: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.seq_scaler: Any = None
        self.xgb_scaler: Any = None
        self.is_loaded = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load feature configs, scalers, and model checkpoints from disk."""
        cfg_path = self.model_dir / "feature_config.json"
        if not cfg_path.exists():
            print(f"Warning: Model artifacts for {self.subset} not found at {self.model_dir}. Please run training first.")
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.feature_config = json.load(f)

        seq_scaler_path = self.model_dir / "scaler.joblib"
        if seq_scaler_path.exists():
            self.seq_scaler = joblib.load(seq_scaler_path)

        xgb_scaler_path = self.model_dir / "xgb_scaler.joblib"
        if xgb_scaler_path.exists():
            self.xgb_scaler = joblib.load(xgb_scaler_path)

        # 1. Load XGBoost if available
        xgb_path = self.model_dir / "best_xgboost.joblib"
        if xgb_path.exists():
            self.models["XGBoost"] = joblib.load(xgb_path)

        # 2. Load PyTorch LSTM if available
        lstm_path = self.model_dir / "best_lstm.pth"
        if lstm_path.exists():
            n_features = len(self.feature_config.get("feature_cols", []))
            lstm_model = SequenceRULModel(cell_type="LSTM", n_features=n_features, hidden_size=64, num_layers=2)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
            lstm_model.to(self.device)
            lstm_model.eval()
            self.models["LSTM"] = lstm_model

        # 3. Load PyTorch GRU if available
        gru_path = self.model_dir / "best_gru.pth"
        if gru_path.exists():
            n_features = len(self.feature_config.get("feature_cols", []))
            gru_model = SequenceRULModel(cell_type="GRU", n_features=n_features, hidden_size=64, num_layers=2)
            gru_model.load_state_dict(torch.load(gru_path, map_location=self.device, weights_only=True))
            gru_model.to(self.device)
            gru_model.eval()
            self.models["GRU"] = gru_model

        self.is_loaded = True
        print(f"InferenceEngine initialized for {self.subset}. Available models: {list(self.models.keys())}")

    def validate_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input dataframe structure, types, and sequence order."""
        if df.empty:
            raise ValueError("Input sensor dataset is empty.")

        data = df.copy()

        # Handle raw headerless files
        if "cycle" not in data.columns and data.shape[1] >= 24:
            data = assign_column_names(data)

        if "cycle" not in data.columns:
            # Auto-synthesize cycle sequence if missing
            data["cycle"] = np.arange(1, len(data) + 1)

        engine_col = "unit_id" if "unit_id" in data.columns else "unit"
        if engine_col not in data.columns:
            data[engine_col] = 1

        # Check numeric types for sensor columns
        feature_cols = self.feature_config.get("feature_cols", [])
        missing_cols = [c for c in feature_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required sensor columns: {missing_cols}")

        for col in feature_cols + ["cycle"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            if data[col].isna().any():
                data[col] = data[col].ffill().bfill().fillna(0.0)

        data = data.sort_values([engine_col, "cycle"]).reset_index(drop=True)
        return data

    def predict(
        self,
        df: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate RUL prediction for the latest observation of the provided engine trajectory."""
        if not self.is_loaded:
            self._load_artifacts()
            if not self.is_loaded:
                raise RuntimeError("Model artifacts are not loaded. Train the models first.")

        data = self.validate_input_df(df)

        available_models = list(self.models.keys())
        if not available_models:
            raise RuntimeError(f"No trained models found in {self.model_dir}")

        if model_name is None:
            model_name = self.feature_config.get("recommended_model", available_models[0])

        if model_name not in self.models:
            # Fallback to first available model
            model_name = available_models[0]

        feature_cols = self.feature_config["feature_cols"]
        seq_len = self.feature_config.get("seq_len", 30)
        n_observed_cycles = len(data)
        current_cycle = int(data["cycle"].iloc[-1])

        # Execute prediction based on model family
        if model_name == "XGBoost":
            tabular_features = self.feature_config["tabular_feature_cols"]
            enriched = extract_tabular_features(data, feature_cols)
            last_row = enriched[tabular_features].iloc[-1:]
            scaled_row = self.xgb_scaler.transform(last_row)
            raw_pred = float(self.models["XGBoost"].predict(scaled_row)[0])
        else:
            # PyTorch Sequence Model (LSTM or GRU)
            sensor_matrix = data[feature_cols].to_numpy(dtype=np.float32)
            window = pad_or_truncate_sequence(sensor_matrix, seq_len)  # (seq_len, n_features)
            # Scale
            F = window.shape[-1]
            scaled_window = self.seq_scaler.transform(window.reshape(-1, F)).reshape(1, seq_len, F)
            tensor_x = torch.from_numpy(scaled_window).float().to(self.device)

            model = self.models[model_name]
            with torch.no_grad():
                out = model(tensor_x)
                raw_pred = float(out.cpu().numpy()[0])

        predicted_rul = max(0.0, float(raw_pred))

        # Maintenance insights & health status
        health_status, guidance, risk_level = self._compute_maintenance_insights(predicted_rul)

        # Extract sensor trends for visualization
        key_sensors = [c for c in ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9", "sensor_11", "sensor_12", "sensor_15"] if c in data.columns]
        trajectory = {
            "cycles": data["cycle"].tolist(),
            "sensors": {s: [round(float(v), 3) for v in data[s].tolist()] for s in key_sensors},
        }

        return {
            "subset": self.subset,
            "model_used": model_name,
            "current_cycle": current_cycle,
            "total_observed_cycles": n_observed_cycles,
            "predicted_rul_cycles": round(predicted_rul, 1),
            "estimated_failure_cycle": int(current_cycle + round(predicted_rul)),
            "health_status": health_status,
            "risk_percentage": risk_level,
            "maintenance_guidance": guidance,
            "trajectory": trajectory,
            "disclaimer": "EngiSense estimates are AI-assisted decision-support indicators, not certified safety assessments or manufacturer-guaranteed failure dates. Always schedule physical engineering inspections.",
        }

    def _compute_maintenance_insights(self, rul: float) -> Tuple[str, str, int]:
        """Categorize engine health into transparent operational categories."""
        if rul < 25.0:
            status = "CRITICAL"
            guidance = (
                f"Urgent Action Recommended: Predicted remaining useful life is only {rul:.1f} cycles. "
                "Schedule immediate borescope and non-destructive inspection (NDI). Remove unit from active service line before next flight cycle to mitigate uncontained failure risk."
            )
            risk = 95
        elif rul < 60.0:
            status = "WARNING"
            guidance = (
                f"Maintenance Planning Advised: Predicted remaining useful life is {rul:.1f} cycles. "
                "Initiate ground maintenance work order. Monitor thermal exhaust gas and spool vibration closely on subsequent flights and prepare replacement hot-section components."
            )
            risk = 60
        else:
            status = "HEALTHY"
            guidance = (
                f"Nominal Operating Condition: Predicted remaining useful life is {rul:.1f} cycles. "
                "Engine parameters are currently within normal baseline envelopes. Continue standard scheduled preventative maintenance protocols."
            )
            risk = 15

        return status, guidance, risk
