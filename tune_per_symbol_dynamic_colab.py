# tune_per_symbol_dynamic_colab.py
from __future__ import annotations
import os
import pickle
import pandas as pd
import optuna
from loguru import logger
import yaml

from src.config import Cfg
from src.data_colab import fetch_bars, merge_features_labels # MODIFIED: Using data_colab
from src.features import build_features, FeatureConfig
from src.labels import binary_up_down
from src.ensemble import Ensemble
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# Removed MetaTrader5 import and initialization

# --- Load config ---
cfg = Cfg.from_yaml("config.yaml")
feat_cfg = FeatureConfig(**cfg.features.__dict__)

# --- Where to store per-symbol best params ---
PARAMS_DIR = "optuna_params"
os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Load YAML for dynamic parameter ranges ---
with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

def suggest_params(trial, model_name, param_ranges: dict):
    """Suggest hyperparameters dynamically from YAML ranges"""
    params = {}
    for k, v in param_ranges.items():
        if isinstance(v, list) and len(v) in [2, 3]:  # [min, max] or [min, max, "log"]
            if len(v) == 3 and v[2] == "log":
                params[k] = trial.suggest_float(f"{model_name}_{k}", v[0], v[1], log=True)
            else:
                if isinstance(v[0], int) and isinstance(v[1], int):
                    params[k] = trial.suggest_int(f"{model_name}_{k}", v[0], v[1])
                else:
                    params[k] = trial.suggest_float(f"{model_name}_{k}", v[0], v[1])
        else:
            params[k] = v  # fixed value
    return params

# --- Objective function for Optuna ---
def objective(trial, symbol: str, data: pd.DataFrame):
    X = data.drop(columns=["y", "close", "high", "low", "volume"])
    y = data["y"]

    # Suggest hyperparameters per model dynamically from YAML
    model_params = {}
    for model in yaml_cfg["models"]:
        model_name = model["name"]
        model_params[model_name] = suggest_params(trial, model_name, model.get("params", {}))

    # Create ensemble with current trial parameters
    ens = Ensemble(cfg, model_params=model_params)
    
    # TimeSeries cross-validation
    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X)//300)))
    aucs = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        ens.fit(X_tr, y_tr)
        p_val = ens.predict_proba(X_val)
        aucs.append(roc_auc_score(y_val, p_val))

    mean_auc = float(pd.Series(aucs).mean())
    return 1 - mean_auc  # Optuna minimizes

# --- Loop over symbols ---
for sym in cfg.symbols:
    logger.info(f"🔹 Tuning models for {sym}...")

    # Fetch historical data & features
    df = fetch_bars(sym, cfg.timeframe, cfg.history_bars)
    X = build_features(df, feat_cfg)
    y = binary_up_down(df, cfg.prediction_horizon)
    data = merge_features_labels(df, X, y)

    # Create and run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, sym, data), n_trials=50)

    # Save best parameters for this symbol
    param_file = os.path.join(PARAMS_DIR, f"{sym.replace('#','')}_best_params.pkl")
    with open(param_file, "wb") as f:
        pickle.dump(study.best_params, f)

    logger.info(f"[{sym}] Best params saved to {param_file}")
    logger.debug(study.best_params)
