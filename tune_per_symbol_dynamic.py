# tune_per_symbol_dynamic.py
from __future__ import annotations
import os
import pickle
import pandas as pd
import optuna
from loguru import logger
import yaml
from functools import partial
from joblib import Parallel, delayed # Added for parallelization
import MetaTrader5 as mt5
from dotenv import load_dotenv # Added
from src.utils import setup_logging # Added

from src.config import Cfg
from src.data import fetch_bars, merge_features_labels # Using src.data
from src.features import build_features, FeatureConfig
from src.labels import binary_up_down
from src.ensemble import Ensemble
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# --- Initial Setup --- # Added
load_dotenv() # Added
setup_logging() # Added

# --- Load config ---
cfg = Cfg.from_yaml("config.yaml")
PARAMS_DIR = "optuna_params"
os.makedirs(PARAMS_DIR, exist_ok=True)

# Load raw YAML for Optuna parameter ranges
with open("config.yaml", "r") as f:
    raw_yaml_cfg = yaml.safe_load(f) # Renamed from yaml_cfg

def suggest_params(trial, prefix: str, param_ranges: dict):
    params = {}
    for k, v in param_ranges.items():
        if isinstance(v, list) and len(v) in [2, 3]:
            if len(v) == 3 and v[2] == "log":
                params[k] = trial.suggest_float(f"{prefix}_{k}", v[0], v[1], log=True)
            else:
                if isinstance(v[0], int) and isinstance(v[1], int):
                    params[k] = trial.suggest_int(f"{prefix}_{k}", v[0], v[1])
                else:
                    params[k] = trial.suggest_float(f"{prefix}_{k}", v[0], v[1])
        else:
            params[k] = v
    return params

def objective(trial, df: pd.DataFrame, y: pd.Series, symbol: str):
    # --- 1. Suggest Feature Parameters ---
    feature_params_raw = suggest_params(trial, "feature", raw_yaml_cfg.get("features", {})) # Changed from yaml_cfg

    # Handle roc_lags specifically as it's a list
    roc_lags_options = [
        (1, 3, 5, 10),
        (1, 2, 4, 8),
        (2, 5, 10, 15),
        (1, 2, 3)
    ]
    if "roc_lags" in feature_params_raw:
        del feature_params_raw["roc_lags"]

    roc_lags_choice = trial.suggest_categorical("feature_roc_lags", roc_lags_options)
    feature_params_raw["roc_lags"] = roc_lags_choice

    feature_cfg = FeatureConfig(**feature_params_raw)

    # --- 2. Build Features for this Trial ---
    X = build_features(df, feature_cfg, symbol)
    data = merge_features_labels(df, X, y)
    
    X_train = data.drop(columns=["y", "close", "high", "low", "volume"])
    y_train = data["y"]

    # --- 3. Suggest Model Hyperparameters ---
    model_params = {}
    for model in yaml_cfg["models"]:
        model_name = model["name"]
        model_params[model_name] = suggest_params(trial, f"model_{model_name}", model.get("params", {}))

    # --- 4. Evaluate Ensemble ---
    ens = Ensemble(cfg, model_params=model_params)
    
    n_splits_calculated = min(5, max(2, len(X_train) // 300))
    logger.debug(f"Calculated n_splits for TimeSeriesSplit: {n_splits_calculated}")
    
    tscv = TimeSeriesSplit(n_splits=n_splits_calculated)
    aucs = []
    for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        ens.fit(X_tr, y_tr)
        p_val = ens.predict_proba(X_val)
        auc = roc_auc_score(y_val, p_val)
        aucs.append(auc)

        # --- Pruning ---
        trial.report(1 - auc, i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_auc = float(pd.Series(aucs).mean())
    return 1 - mean_auc  # Optuna minimizes

def run_tuning_for_symbol(sym: str):
    # Initialize MT5 for this process
    if not mt5.initialize():
        logger.error(f"MetaTrader5 initialization failed for {sym}. Skipping.")
        return

    logger.info(f"🔹 Starting combined feature and model tuning for {sym}...")

    # --- Pre-load data once per symbol ---
    logger.info(f"Fetching data for {sym}...")
    df = fetch_bars(sym, cfg.timeframe, cfg.history_bars)
    y = binary_up_down(df, cfg.prediction_horizon)
    
    # --- Create a partial function for the objective ---
    objective_partial = partial(objective, df=df, y=y, symbol=sym)

    study_name = f"feature_model_tuning_{sym.replace('#','')}_history_{cfg.history_bars}"
    storage_path = f"sqlite:///{PARAMS_DIR}/{study_name}.db"
    
    # --- Add a pruner to the study ---
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_path, load_if_exists=True, pruner=pruner)
    
    study.optimize(objective_partial, n_trials=10) # Using 10 trials as in colab version

    # --- Process and Save Best Parameters ---
    best_params_flat = study.best_params
    best_params_structured = {"features": {}, "models": {}}

    for key, value in best_params_flat.items():
        if key.startswith("feature_"):
            param_name = key.replace("feature_", "")
            best_params_structured["features"][param_name] = value
        elif key.startswith("model_"):
            parts = key.split('_')
            model_name = parts[1]
            param_name = '_'.join(parts[2:])
            if model_name not in best_params_structured["models"]:
                best_params_structured["models"][model_name] = {}
            best_params_structured["models"][model_name][param_name] = value

    param_file = os.path.join(PARAMS_DIR, f"{sym.replace('#','')}_best_params.pkl")
    with open(param_file, "wb") as f:
        pickle.dump(best_params_structured, f)

    logger.info(f"[{sym}] Best combined params saved to {param_file}")
    logger.debug(best_params_structured)
    
    # Shutdown MT5 for this process
    mt5.shutdown()

# --- Main Execution Loop (Parallelized) ---
if __name__ == '__main__':
    # Use all available CPU cores (-1) or specify a number (e.g., 2 for your system)
    Parallel(n_jobs=-1)(delayed(run_tuning_for_symbol)(sym) for sym in cfg.symbols)
