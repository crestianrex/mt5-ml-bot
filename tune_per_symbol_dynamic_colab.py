# tune_per_symbol_dynamic_colab.py
from __future__ import annotations
import os
import pickle
import pandas as pd
import optuna
from loguru import logger
import yaml
from functools import partial
from joblib import Parallel, delayed
import traceback # Added for detailed error logging

from src.config import Cfg
from src.data_colab import fetch_bars, merge_features_labels
from src.features import build_static_features, build_dynamic_features, FeatureConfig
from src.labels import binary_up_down
from src.ensemble import Ensemble
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# --- Detect Colab and set path ---
# Assumes drive is already mounted if running in Colab.
try:
    import google.colab
    # This path should point to the location in your Google Drive where params are stored.
    PARAMS_DIR = "/content/drive/MyDrive/mt5_ml_bot_params/optuna_params"
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    PARAMS_DIR = "optuna_params"

os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Load config ---
cfg = Cfg.from_yaml("config.yaml")

with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

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

def objective(trial, df: pd.DataFrame, y: pd.Series, static_features: pd.DataFrame, symbol: str):
    try:
        # --- 1. Suggest Feature Parameters ---
        feature_params_raw = suggest_params(trial, "feature", yaml_cfg.get("features", {}))

        roc_lags_options_raw = yaml_cfg.get("roc_lags_options", [
            [1, 3, 5, 10], [1, 2, 4, 8], [2, 5, 10, 15], [1, 2, 3]
        ])
        # Convert lists to tuples for Optuna's categorical choice
        roc_lags_options = [tuple(l) for l in roc_lags_options_raw]

        if "roc_lags" in feature_params_raw:
            del feature_params_raw["roc_lags"]

        roc_lags_choice = trial.suggest_categorical("feature_roc_lags", roc_lags_options)
        feature_params_raw["roc_lags"] = roc_lags_choice

        feature_cfg = FeatureConfig(**feature_params_raw)

        # --- 2. Build Features for this Trial (using cached static features) ---
        X = build_dynamic_features(df, static_features, feature_cfg, symbol)
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
        
        cv_samples_per_split = yaml_cfg.get("cv_samples_per_split", 300)
        n_splits_calculated = min(5, max(2, len(X_train) // cv_samples_per_split))
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

            trial.report(1 - auc, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_auc = float(pd.Series(aucs).mean())
        return 1 - mean_auc

    except optuna.exceptions.TrialPruned as e:
        raise e # Allow Optuna to handle pruning
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"--- Trial Failed ---\nError: {e}\nTraceback:\n{tb_str}")
        return float('inf')

def run_tuning_for_symbol(sym: str):
    logger.info(f"🔹 Starting combined feature and model tuning for {sym}...")

    logger.info(f"Fetching data for {sym}...")
    df = fetch_bars(sym, cfg.timeframe, cfg.history_bars)
    y = binary_up_down(df, cfg.prediction_horizon)
    
    # --- Pre-calculate and cache static features ---
    static_features = build_static_features(df, symbol=sym)
    
    objective_partial = partial(objective, df=df, y=y, static_features=static_features, symbol=sym)

    study_name = f"feature_model_tuning_{sym.replace('#','_')}_history_{cfg.history_bars}"
    storage_path = f"sqlite:///{os.path.join(PARAMS_DIR, study_name)}.db"
    
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_path, load_if_exists=True, pruner=pruner)
    
    n_trials = yaml_cfg.get("optuna_n_trials", 100)
    study.optimize(objective_partial, n_trials=n_trials)

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

    param_file = os.path.join(PARAMS_DIR, f"{sym.replace('#','_')}_best_params.pkl")
    with open(param_file, "wb") as f:
        pickle.dump(best_params_structured, f)

    logger.info(f"[{sym}] Best combined params saved to {param_file}")
    logger.debug(best_params_structured)
    
    # --- Generate and Save Visualization Plots ---
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(PARAMS_DIR, f"{study_name}_optimization_history.png"))
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(PARAMS_DIR, f"{study_name}_param_importances.png"))
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Could not generate plots. Make sure you have 'plotly' and 'kaleido' installed. Error: {e}")


# --- Main Execution Loop (Parallelized) ---
if __name__ == '__main__':
    n_jobs = yaml_cfg.get("n_jobs", -1)
    Parallel(n_jobs=n_jobs)(delayed(run_tuning_for_symbol)(sym) for sym in cfg.symbols)