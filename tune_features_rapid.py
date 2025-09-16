# tune_features_rapid.py
# This script performs a rapid, focused optimization on feature parameters only.
from __future__ import annotations
import os
import pickle
import pandas as pd
import optuna
from loguru import logger
import yaml

from src.config import Cfg
from src.data_colab import fetch_bars, merge_features_labels
from src.features import build_features, FeatureConfig
from src.labels import binary_up_down
from src.strategy_ml import MLStrategy # Using single model instead of full ensemble
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# --- Load config ---
cfg = Cfg.from_yaml("config.yaml")
PARAMS_DIR = "optuna_params"
os.makedirs(PARAMS_DIR, exist_ok=True)

with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

def suggest_feature_params(trial, param_ranges: dict):
    params = {}
    # We are only tuning a subset of features for speed
    tunable_features = ["adx_period", "adx_trend_thresh"]
    for k, v in param_ranges.items():
        if k in tunable_features:
            if isinstance(v, list) and len(v) == 2:
                params[k] = trial.suggest_int(f"feature_{k}", v[0], v[1])
            else:
                params[k] = v # Use default if not a range
        else:
            # Keep other feature params fixed to their default
            if isinstance(v, list):
                params[k] = v[0] # Use the first value as default if it's a range
            else:
                params[k] = v
    return params

def objective(trial, symbol: str):
    # --- 1. Suggest Feature Parameters ---
    feature_params_raw = suggest_feature_params(trial, yaml_cfg.get("features", {}))
    feature_cfg = FeatureConfig(**feature_params_raw)

    # --- 2. Build Features for this Trial ---
    df = fetch_bars(symbol, cfg.timeframe, cfg.history_bars)
    X = build_features(df, feature_cfg, symbol)
    y = binary_up_down(df, cfg.prediction_horizon)
    data = merge_features_labels(df, X, y)
    
    X_train = data.drop(columns=["y", "close", "high", "low", "volume"])
    y_train = data["y"]

    # --- 3. Evaluate with a single fast model (Logistic Regression) ---
    model = MLStrategy(model='logreg', calibrate=False)
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for tr_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        p_val = model.predict_proba(X_val)
        aucs.append(roc_auc_score(y_val, p_val))

    mean_auc = float(pd.Series(aucs).mean())
    return 1 - mean_auc  # Optuna minimizes

# --- Main Execution Loop ---
for sym in cfg.symbols:
    if sym != 'EURJPY#':
        logger.info(f"Skipping tuning for {sym} as per current focus.")
        continue

    logger.info(f"🔹 Starting RAPID FEATURE tuning for {sym}...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, sym), n_trials=25)

    # --- Process and Save Best Feature Parameters ---
    best_feature_params = study.best_params
    best_feature_params_clean = {k.replace('feature_', ''): v for k, v in best_feature_params.items()}

    param_file = os.path.join(PARAMS_DIR, f"{sym.replace('#','')}_best_features.pkl")
    with open(param_file, "wb") as f:
        pickle.dump(best_feature_params_clean, f)

    logger.info(f"[{sym}] Best feature params saved to {param_file}")
    logger.info(f"Found best feature params: {best_feature_params_clean}")
