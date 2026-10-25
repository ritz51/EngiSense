"""Comprehensive test suite for EngiSense.

Verifies:
- Dataset loading and column validation
- RUL target calculations and clipping
- Engine-based splitting and leakage prevention
- Feature selection and preprocessing consistency
- Sequence-window construction and target alignment
- Model loading and inference execution
- Metric calculations (MAE, RMSE, R2, NASA score)
- Invalid input handling
- FastAPI backend integration endpoints
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data_loader import load_dataset, load_rul_file, load_subset, assign_column_names
from src.rul import compute_rul_train, compute_rul_test, clip_rul
from src.metrics import compute_metrics, compute_nasa_score
from src.feature_engineering import (
    get_feature_columns,
    split_engine_ids,
    extract_tabular_features,
    pad_or_truncate_sequence,
    build_sequence_dataset,
    CONSTANT_FEATURES_MAP,
)
from src.inference import InferenceEngine
from src.api import app


# ---------------------------------------------------------------------------
# 1. Dataset Loading & Column Validation Tests
# ---------------------------------------------------------------------------

def test_load_dataset_fd001():
    train_df = load_dataset("FD001", "train")
    assert not train_df.empty
    assert train_df.shape[0] == 20631
    assert train_df.shape[1] == 26
    assert "unit_id" in train_df.columns
    assert "cycle" in train_df.columns
    assert "setting_1" in train_df.columns
    assert "sensor_21" in train_df.columns

    test_df = load_dataset("FD001", "test")
    assert not test_df.empty
    assert test_df.shape[0] == 13096
    assert test_df.shape[1] == 26


def test_load_rul_file_fd001():
    rul_series = load_rul_file("FD001")
    assert len(rul_series) == 100
    assert (rul_series > 0).all()
    assert np.issubdtype(rul_series.dtype, np.integer)


def test_load_dataset_invalid_parameters():
    with pytest.raises(ValueError):
        load_dataset("FD001", "invalid_kind")

    with pytest.raises(FileNotFoundError):
        load_dataset("FD999", "train")


# ---------------------------------------------------------------------------
# 2. RUL Formulation Tests
# ---------------------------------------------------------------------------

def test_compute_rul_train():
    df = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "sensor_1": [10.0, 10.0, 10.0, 10.0, 10.0],
    })
    result = compute_rul_train(df)
    assert "RUL" in result.columns
    # Engine 1 runs cycles 1, 2, 3 -> RUL 2, 1, 0
    # Engine 2 runs cycles 1, 2 -> RUL 1, 0
    assert result["RUL"].tolist() == [2, 1, 0, 1, 0]


def test_compute_rul_test():
    df = pd.DataFrame({
        "unit_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 3],
    })
    # Engine 1 last cycle is 2, ref RUL is 10 -> RUL at cycle 2 is 10, cycle 1 is 11
    # Engine 2 last cycle is 3, ref RUL is 20 -> RUL at cycle 3 is 20, cycle 1 is 22
    ref_ruls = np.array([10, 20])
    result = compute_rul_test(df, ref_ruls)
    assert result["RUL"].tolist() == [11, 10, 22, 20]


def test_clip_rul():
    df = pd.DataFrame({"RUL": [150, 125, 100, 20, 0]})
    clipped = clip_rul(df, max_rul=125)
    assert clipped["RUL"].tolist() == [125, 125, 100, 20, 0]


# ---------------------------------------------------------------------------
# 3. Leakage Prevention & Engine-Based Splitting Tests
# ---------------------------------------------------------------------------

def test_split_engine_ids_disjoint():
    df = pd.DataFrame({
        "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "cycle": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    })
    train_ids, val_ids = split_engine_ids(df, val_frac=0.4, seed=42)
    # Sets must be strictly disjoint (no engine appears in both)
    assert set(train_ids).isdisjoint(set(val_ids))
    assert len(train_ids) + len(val_ids) == 5


# ---------------------------------------------------------------------------
# 4. Feature Selection & Sequence Processing Tests
# ---------------------------------------------------------------------------

def test_feature_selection_fd001():
    train_df = load_dataset("FD001", "train")
    features = get_feature_columns(train_df, subset="FD001", drop_constants=True)
    # Constant sensors should not be present
    for const_col in ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19", "setting_3"]:
        assert const_col not in features
    # Active sensors must be present
    assert "sensor_2" in features
    assert "sensor_3" in features
    assert "sensor_4" in features
    assert len(features) == 17


def test_pad_or_truncate_sequence():
    # Sequence shorter than required seq_len (e.g. 5 rows, target 10)
    data = np.arange(15).reshape(5, 3)
    padded = pad_or_truncate_sequence(data, seq_len=10)
    assert padded.shape == (10, 3)
    # The first row is repeated for edge-padding
    assert np.array_equal(padded[0], data[0])
    assert np.array_equal(padded[-1], data[-1])

    # Sequence longer than required seq_len (e.g. 15 rows, target 10)
    longer = np.arange(45).reshape(15, 3)
    truncated = pad_or_truncate_sequence(longer, seq_len=10)
    assert truncated.shape == (10, 3)
    assert np.array_equal(truncated[-1], longer[-1])


def test_tabular_rolling_features():
    df = pd.DataFrame({
        "unit_id": [1, 1, 1, 1, 1],
        "cycle": [1, 2, 3, 4, 5],
        "sensor_2": [10.0, 12.0, 14.0, 16.0, 18.0],
    })
    enriched = extract_tabular_features(df, feature_cols=["sensor_2"], rolling_windows=[3])
    assert "sensor_2_roll_mean_3" in enriched.columns
    assert "sensor_2_roll_std_3" in enriched.columns
    # Mean of cycle 3 (window 3: 10, 12, 14) is 12.0
    assert np.isclose(enriched["sensor_2_roll_mean_3"].iloc[2], 12.0)


# ---------------------------------------------------------------------------
# 5. Metrics & NASA Asymmetric Scoring Tests
# ---------------------------------------------------------------------------

def test_metrics_calculation():
    y_true = np.array([100.0, 50.0, 20.0])
    y_pred = np.array([100.0, 50.0, 20.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0
    assert metrics["NASA_Score"] == 0.0


def test_nasa_asymmetric_property():
    # Under-predicting (early prediction d = -10): safer, smaller penalty
    score_early = compute_nasa_score(np.array([50.0]), np.array([40.0]))
    # Over-predicting (late prediction d = +10): dangerous, larger penalty
    score_late = compute_nasa_score(np.array([50.0]), np.array([60.0]))
    assert score_late > score_early


# ---------------------------------------------------------------------------
# 6. Inference Engine & Backend API Tests
# ---------------------------------------------------------------------------

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "subsets_supported" in data


def test_api_overview():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "subsets" in data
    assert "constant_sensors_fd001" in data


def test_api_models():
    response = client.get("/api/models?subset=FD001")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "XGBoost" in data["models"]
    assert "LSTM" in data["models"]
    assert "GRU" in data["models"]
    assert data["recommended_model"] in ["XGBoost", "LSTM", "GRU"]


def test_api_engines_list():
    response = client.get("/api/engines?subset=FD001")
    assert response.status_code == 200
    data = response.json()
    assert data["total_test_engines"] == 100
    assert len(data["engines"]) == 100


def test_api_engine_trajectory():
    response = client.get("/api/engine/FD001/1")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 1
    assert data["total_cycles"] == 31
    assert data["official_reference_rul"] == 112
    assert "sensors" in data
    assert "sensor_2" in data["sensors"]


def test_api_predict_test_engine():
    response = client.post("/api/predict", json={
        "subset": "FD001",
        "engine_id": 1,
        "model_name": "XGBoost",
    })
    assert response.status_code == 200
    res = response.json()
    assert res["engine_id"] == 1
    assert res["predicted_rul_cycles"] >= 0
    assert res["official_reference_rul"] == 112
    assert res["health_status"] in ["HEALTHY", "WARNING", "CRITICAL"]
    assert "maintenance_guidance" in res


def test_api_predict_invalid_engine():
    response = client.post("/api/predict", json={
        "subset": "FD001",
        "engine_id": 9999,
    })
    assert response.status_code == 404


def test_api_predict_invalid_body():
    response = client.post("/api/predict", json={
        "subset": "FD001",
    })
    assert response.status_code == 400
