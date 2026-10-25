# EngiSense — Smart Engine Health Monitoring

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-1182c5.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)]()

**EngiSense** is an industrial-grade Remaining Useful Life (RUL) prediction and turbofan health analytics system built on the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation dataset.

The system performs automated validation, statistical exploratory data analysis, feature engineering (including rolling temporal dynamics and constant sensor elimination), and trains and evaluates three distinct machine learning architectures: **XGBoost baseline**, **PyTorch LSTM**, and **PyTorch GRU**. Inferences and maintenance decision-support insights are served in real time via a **FastAPI** backend and an industrial **dashboard interface**.

---

## Architecture Overview

```
                      NASA C-MAPSS Datasets (FD001–FD004)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       Data Loading & Validation               Statistical Inference & EDA
      (Missing/Inf/Duplicates Check,         (Engine Lifetime ANOVA/Kruskal,
     Constant Sensor Identification)         Benjamini-Hochberg FDR Corr,
                 │                           Mixed-Effects Trend Models)
                 ▼                                       │
     Leakage-Free Preprocessing                          │
   (Engine-Level Splits, Scaling,                        │
     Sequence Windows, Rolling Stats)                    │
                 │                                       │
       ┌─────────┴─────────┐                             │
       ▼                   ▼                             │
  Tabular Features   Sequence Windows                    │
       │                   │                             │
       ▼                   ▼                             │
    XGBoost            PyTorch                           │
   Regressor          LSTM / GRU                         │
       │                   │                             │
       └─────────┬─────────┘                             │
                 ▼                                       │
      Quantitative Evaluation                            │
  (Test RMSE, MAE, R², NASA Score,                       │
   Error Distributions, Diagnostic Plots)                │
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                          FastAPI Backend Service
                       (REST API & Artifact Serving)
                                     │
                                     ▼
                       EngiSense Web Application
                 (Fleet Dashboard, Real-time RUL Gauge,
               Sensor Trajectories, Maintenance Guidance)
```

---

## Measured Benchmark Performance (FD001 Official Test Set)

All models were evaluated on the **official held-out NASA test set** (100 test engines) using ground-truth reference RUL labels (`RUL_FD001.txt`).

| Model Architecture | Input Representation | Test RMSE (Cycles) | Test MAE (Cycles) | Test R² Score | NASA Asymmetric Penalty | Training Time | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch LSTM** | 30-Cycle Sequence Window | **15.76** | **11.06** | **0.8562** | 618.42 | 57.5s | **Recommended Best Model** |
| **PyTorch GRU** | 30-Cycle Sequence Window | **16.12** | **11.77** | **0.8496** | **456.52** | 152.5s | Lowest NASA Penalty |
| **XGBoost** | Tabular Sensors + Rolling Dynamics | **19.37** | **14.09** | **0.7827** | 1372.26 | 1.5s | Fast Baseline |

> **Selection Rationale:** The **PyTorch LSTM** achieved the lowest overall Test RMSE (15.76 cycles) and highest R² (0.8562), making it the primary recommended model for high-precision cycle-by-cycle predictions. The **PyTorch GRU** achieved the lowest NASA asymmetric penalty score (456.52), reflecting favorable penalties against catastrophic late failures.

---

## Repository Structure

```
EngiSense-team/
├── archive/                         # NASA C-MAPSS raw dataset files
│   ├── train_FD001.txt ... FD004.txt
│   ├── test_FD001.txt ... FD004.txt
│   ├── RUL_FD001.txt ... FD004.txt
│   └── readme.txt                   # NASA dataset documentation
├── frontend/                        # Industrial Dashboard Frontend
│   ├── index.html                   # Semantic HTML5 UI layout
│   ├── styles.css                   # Custom CSS variables, dark industrial theme
│   └── app.js                       # Chart.js integration, live inference controller
├── outputs/                         # Generated artifacts and evaluation outputs
│   ├── models/FD001/
│   │   ├── best_lstm.pth            # Trained PyTorch LSTM checkpoint
│   │   ├── best_gru.pth             # Trained PyTorch GRU checkpoint
│   │   ├── best_xgboost.joblib      # Trained XGBoost model
│   │   ├── scaler.joblib            # Sequence StandardScaler
│   │   ├── xgb_scaler.joblib        # Tabular StandardScaler
│   │   ├── feature_config.json      # Feature selection & metadata configuration
│   │   ├── model_comparison.json    # Full comparison metrics and parameters
│   │   ├── actual_vs_predicted.png  # True vs predicted RUL scatter comparison
│   │   ├── residual_analysis.png    # Error distribution density curves
│   │   ├── model_comparison.png     # Side-by-side metric bar charts
│   │   ├── training_loss.png        # Sequence model training/validation loss curves
│   │   └── feature_importance_xgboost.png # Top 15 XGBoost feature importances
│   ├── rul/                         # RUL validation distributions and summaries
│   │   ├── FD001_train_summary.csv
│   │   └── rul_distribution_*.png
│   ├── validation_report.json       # Dataset validation output (JSON)
│   ├── validation_report.csv        # Dataset validation output (CSV)
│   └── statistical_inference_report.json # Full statistical inference report
├── src/                             # Backend and Core Python Modules
│   ├── __init__.py
│   ├── api.py                       # FastAPI application & REST endpoints
│   ├── data_loader.py               # Robust whitespace C-MAPSS dataset loader
│   ├── feature_engineering.py       # Rolling statistics, constant filtering, sequence windows
│   ├── inference.py                 # Real-time InferenceEngine with maintenance logic
│   ├── metrics.py                   # MAE, RMSE, R2, and official NASA scoring function
│   ├── rul.py                       # RUL target formulations and distribution plotting
│   ├── statistical_inference.py     # ANOVA, FDR correction, mixed-effects regression
│   ├── train_models.py              # Unified trainer for XGBoost, LSTM, and GRU
│   ├── train_rul.py                 # PyTorch sequence training pipeline
│   ├── validate_rul.py              # RUL sanity and verification runner
│   └── validation.py                # Dataset validation and constant sensor detection
├── tests/                           # Automated Pytest Suite
│   ├── __init__.py
│   └── test_pipeline.py             # 20 automated unit and integration tests
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## Installation & Windows Environment Setup

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.12 64-bit on Windows)
- **PowerShell** or Command Prompt

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## Step-by-Step Usage & Commands

### 1. Dataset Validation
Validates dimensions, duplicate rows, missing values, non-numeric values, and identifies constant sensors across all subsets:
```powershell
python -m src.validation
```
*Outputs: `outputs/validation_report.json` and `outputs/validation_report.csv`*

### 2. RUL Formulation & Distribution Verification
Verifies training RUL (cycles to failure) and test RUL calculations (aligned with NASA official references):
```powershell
python -m src.validate_rul
```
*Outputs: Distribution plots and summary CSVs in `outputs/rul/`*

### 3. Exploratory Data Analysis & Statistical Inference
Runs engine lifetime comparisons (ANOVA / Kruskal-Wallis with bootstrap CIs), Benjamini-Hochberg FDR-adjusted sensor-RUL correlations, and random-intercept mixed-effects trend models:
```powershell
python -m src.statistical_inference
```
*Outputs: `outputs/statistical_inference_report.json`*

### 4. Train and Compare Models (XGBoost, LSTM, GRU)
Trains all three architectures on FD001, calculates validation and official test metrics, saves models, feature configs, and diagnostic plots:
```powershell
python -m src.train_models --subset FD001 --epochs 30 --seq_len 30 --max_rul 125
```
*Outputs: All model checkpoints, scalers, and plots in `outputs/models/FD001/`*

### 5. Run Automated Tests
Executes the comprehensive test suite covering data loaders, RUL computation, feature engineering, sequence generation, metrics, inference, and API endpoints:
```powershell
python -m pytest -v tests/
```

### 6. Launch Application (FastAPI & Dashboard UI)
Start the backend web server:
```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

---

## Key Features & Capabilities

### 1. Interactive Fleet Dashboard
- Real-time KPI summaries: Total fleet trajectories, Best Test RMSE, Best Test MAE, and NASA Score.
- Fleet health distribution chart categorizing engines into **Healthy**, **Maintenance Warning**, and **Critical Urgent**.
- Side-by-side quantitative model comparison table.

### 2. Engine RUL Prediction & Diagnostics
- **Test Engine Selector**: Select any of the 100 official test engines from FD001.
- **Custom File Upload**: Upload custom engine sensor logs in standard C-MAPSS text or CSV format.
- **Model Selector**: Switch between PyTorch LSTM, PyTorch GRU, and XGBoost in real time.
- **Dynamic RUL Gauge**: Color-coded circular gauge displaying remaining cycles and estimated end-of-life cycle.
- **Ground Truth Comparison**: Displays official NASA benchmark RUL vs model prediction and exact cycle error.
- **Operational Maintenance Guidance**: Cautious, action-oriented engineering guidance based on configurable degradation thresholds.
- **Live Sensor Trajectory Chart**: Interactive multi-sensor trend visualization (Sensor 2, 3, 4, 7, 9, 11, etc.) plotted over operating cycles.

### 3. Model Benchmark & Visual Diagnostics Gallery
- Comprehensive comparison table of Validation vs Test metrics.
- Interactive Chart.js accuracy and penalty comparison charts.
- Diagnostic artifact viewer for:
  - Actual vs Predicted RUL scatter alignment ($y = x$)
  - Error residual density distributions
  - Sequence model training & validation loss curves
  - XGBoost feature importance rankings

### 4. Statistical Analysis (EDA) Tab
- Hypothesis testing for observed engine lifetimes across subsets.
- Ranked sensor correlations with Remaining Useful Life with FDR $p$-value adjustment.
- Mixed-effects regression slopes ($\beta_1$) and Intraclass Correlation Coefficients (ICC) quantifying cycle degradation rates while controlling for repeated engine measurements.

### 5. Dataset & Sensor Metadata Inspector
- Complete 21-sensor measurement legend with physical descriptions and units.
- Documented feature-selection decisions (filtering non-informative constant sensors in FD001).

---

## Maintenance Guidance & Safety Disclaimer

EngiSense classifies engine health into three operational decision-support categories:

- **HEALTHY (RUL $\ge$ 60 cycles):** Operating parameters are within nominal envelopes. Continue standard scheduled preventative maintenance protocols.
- **WARNING (25 $\le$ RUL $<$ 60 cycles):** Elevated wear detected. Initiate ground maintenance work order, monitor exhaust gas temperatures and spool vibration, and stage replacement hot-section components.
- **CRITICAL (RUL $<$ 25 cycles):** Immediate action required. Schedule borescope and non-destructive inspection (NDI). Remove unit from active service prior to next flight cycle to mitigate failure risk.

> **Disclaimer:** Predictions generated by EngiSense are machine-learning decision-support indicators designed to assist maintenance planning. They do not constitute certified safety assessments, airworthiness releases, or manufacturer-guaranteed failure dates. Always follow OEM technical manuals and physical inspection procedures.
