"""FastAPI backend service for EngiSense.

Exposes REST API endpoints for:
- System health and configuration
- Dataset summaries and validation reports
- Model evaluation metrics and comparison reports
- Statistical inference results
- Sample engine trajectory inspection
- Real-time RUL prediction on test engines or uploaded sensor logs
- Static file serving for the EngiSense frontend dashboard
"""

from __future__ import annotations

import json
import io
import pathlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd

from .data_loader import load_subset, PROJECT_ROOT
from .inference import InferenceEngine

app = FastAPI(
    title="EngiSense API",
    description="Smart Engine Health Monitoring & Remaining Useful Life (RUL) Prediction API",
    version="1.0.0",
)

# Enable CORS for local development and browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine cache for inference
inference_engines: Dict[str, InferenceEngine] = {}

def get_engine(subset: str = "FD001") -> InferenceEngine:
    subset = subset.upper()
    if subset not in inference_engines:
        inference_engines[subset] = InferenceEngine(subset=subset)
    return inference_engines[subset]


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    subset: str = "FD001"
    engine_id: Optional[int] = None
    model_name: Optional[str] = None
    # Optional raw sensor readings as list of dicts
    sensor_data: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Return system status and active models."""
    engine = get_engine("FD001")
    return {
        "status": "healthy",
        "service": "EngiSense — Smart Engine Health Monitoring",
        "version": "1.0.0",
        "subsets_supported": ["FD001", "FD002", "FD003", "FD004"],
        "active_models": list(engine.models.keys()),
        "recommended_model": engine.feature_config.get("recommended_model", "XGBoost"),
    }


@app.get("/api/overview")
def get_overview() -> Dict[str, Any]:
    """Return dataset summaries, engine counts, and validation checks."""
    val_report_path = PROJECT_ROOT / "outputs" / "validation_report.json"
    if val_report_path.exists():
        with open(val_report_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
    else:
        val_data = []

    return {
        "dataset_name": "NASA C-MAPSS Turbofan Degradation",
        "subsets": ["FD001", "FD002", "FD003", "FD004"],
        "validation_summary": val_data,
        "constant_sensors_fd001": ["setting_3", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"],
        "active_sensors_fd001": ["sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21", "setting_1", "setting_2"],
    }


@app.get("/api/models")
def get_model_metrics(subset: str = Query("FD001")) -> Dict[str, Any]:
    """Return actual measured model metrics (MAE, RMSE, R2, NASA score) for XGBoost, LSTM, and GRU."""
    subset = subset.upper()
    comp_path = PROJECT_ROOT / "outputs" / "models" / subset / "model_comparison.json"
    if not comp_path.exists():
        raise HTTPException(status_code=404, detail=f"Model comparison report not found for {subset}. Models must be trained first.")

    with open(comp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/stats")
def get_statistical_inference() -> Dict[str, Any]:
    """Return statistical inference results (ANOVA, sensor correlations, mixed-effects trends)."""
    stat_path = PROJECT_ROOT / "outputs" / "statistical_inference_report.json"
    if not stat_path.exists():
        raise HTTPException(status_code=404, detail="Statistical inference report not found.")

    with open(stat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/engines")
def list_test_engines(subset: str = Query("FD001")) -> Dict[str, Any]:
    """List available test engines with observed cycle counts and official reference RULs."""
    subset = subset.upper()
    try:
        _, test_df, rul_series = load_subset(subset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading subset {subset}: {e}")

    engine_col = "unit_id" if "unit_id" in test_df.columns else "unit"
    engine_groups = test_df.groupby(engine_col)["cycle"].max()

    engines = []
    for idx, (eng_id, max_c) in enumerate(engine_groups.items()):
        true_rul = int(rul_series.iloc[idx]) if idx < len(rul_series) else None
        engines.append({
            "engine_id": int(eng_id),
            "observed_cycles": int(max_c),
            "official_reference_rul": true_rul,
        })

    return {
        "subset": subset,
        "total_test_engines": len(engines),
        "engines": engines,
    }


@app.get("/api/engine/{subset}/{engine_id}")
def get_engine_trajectory(subset: str, engine_id: int) -> Dict[str, Any]:
    """Return full cycle-by-cycle trajectory for a specific engine."""
    subset = subset.upper()
    try:
        _, test_df, rul_series = load_subset(subset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading subset {subset}: {e}")

    engine_col = "unit_id" if "unit_id" in test_df.columns else "unit"
    eng_data = test_df[test_df[engine_col] == engine_id].sort_values("cycle")

    if eng_data.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found in test set for {subset}.")

    # Key informative sensor names
    key_sensors = [c for c in ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9", "sensor_11", "sensor_12", "sensor_15"] if c in eng_data.columns]

    # Find official test RUL for this engine
    unique_ids = test_df[engine_col].drop_duplicates().tolist()
    ref_rul = None
    if engine_id in unique_ids:
        pos = unique_ids.index(engine_id)
        if pos < len(rul_series):
            ref_rul = int(rul_series.iloc[pos])

    return {
        "subset": subset,
        "engine_id": engine_id,
        "total_cycles": len(eng_data),
        "official_reference_rul": ref_rul,
        "cycles": eng_data["cycle"].tolist(),
        "sensors": {s: [round(float(v), 3) for v in eng_data[s].tolist()] for s in key_sensors},
    }


@app.post("/api/predict")
def predict_rul(payload: PredictRequest) -> Dict[str, Any]:
    """Execute real model RUL prediction for an engine ID or provided sensor rows."""
    engine = get_engine(payload.subset)

    # Case 1: Engine ID selected from dataset
    if payload.engine_id is not None:
        try:
            _, test_df, rul_series = load_subset(payload.subset)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed loading subset {payload.subset}: {e}")

        engine_col = "unit_id" if "unit_id" in test_df.columns else "unit"
        eng_data = test_df[test_df[engine_col] == payload.engine_id].sort_values("cycle")
        if eng_data.empty:
            raise HTTPException(status_code=404, detail=f"Engine {payload.engine_id} not found.")

        unique_ids = test_df[engine_col].drop_duplicates().tolist()
        official_rul = None
        if payload.engine_id in unique_ids:
            pos = unique_ids.index(payload.engine_id)
            if pos < len(rul_series):
                official_rul = int(rul_series.iloc[pos])

        result = engine.predict(eng_data, model_name=payload.model_name)
        result["engine_id"] = payload.engine_id
        result["official_reference_rul"] = official_rul
        if official_rul is not None:
            result["prediction_error_cycles"] = round(result["predicted_rul_cycles"] - official_rul, 1)

        return result

    # Case 2: Provided raw sensor table
    elif payload.sensor_data is not None and len(payload.sensor_data) > 0:
        df = pd.DataFrame(payload.sensor_data)
        result = engine.predict(df, model_name=payload.model_name)
        result["engine_id"] = "Uploaded Data"
        return result

    else:
        raise HTTPException(status_code=400, detail="Must provide either an engine_id or sensor_data.")


@app.post("/api/upload_predict")
async def upload_and_predict(
    file: UploadFile = File(...),
    subset: str = Form("FD001"),
    model_name: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload a sensor data file (CSV or whitespace-separated C-MAPSS text) and return real RUL prediction."""
    engine = get_engine(subset)
    contents = await file.read()

    try:
        text = contents.decode("utf-8")
        # Try whitespace or comma delimiter
        first_line = text.strip().split("\n")[0]
        sep = "," if "," in first_line else r"\s+"
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None if "," not in first_line else "infer", engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {e}")

    result = engine.predict(df, model_name=model_name)
    result["engine_id"] = file.filename
    return result


@app.get("/api/plots/{subset}/{filename}")
def get_model_plot(subset: str, filename: str):
    """Serve saved model evaluation and diagnostic visualization plots."""
    subset = subset.upper()
    plot_path = PROJECT_ROOT / "outputs" / "models" / subset / filename
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot {filename} not found for {subset}.")
    return FileResponse(plot_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Serve Frontend Static Files
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
