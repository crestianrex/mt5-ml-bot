# tuner.py
from __future__ import annotations

import os
import pickle
import yaml
import pandas as pd
import optuna
from loguru import logger
from functools import partial
from joblib import Parallel, delayed
import traceback

from src.config import Cfg
from src.data_colab import fetch_bars, merge_features_labels
from src.features import build_static_features, build_dynamic_features, build_features, FeatureConfig
from src.labels import binary_up_down
from src.ensemble import Ensemble
from src.strategy_ml import MLStrategy  # used for Stage 1 (lightweight)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# --- Config ---
cfg = Cfg.from_yaml("config.yaml")
with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

PARAMS_DIR = os.path.join("optuna_params", "hybrid")
os.makedirs(PARAMS_DIR, exist_ok=True)

# --- Stage 1: Rapid Feature Tuning ---
def suggest_feature_params(trial, param_ranges: dict):
    params = {}
    tunable_features = ["adx_period", "adx_trend_thresh"]
    for k, v in param_ranges.items():
        if k in tunable_features:
            if isinstance(v, list) and len(v) == 2:
                params[k] = trial.suggest_int(f"feature_{k}", v[0], v[1])
            else:
                params[k] = v
        else:
            if isinstance(v, list):
                params[k] = v[0]  # default to first value
            else:
                params[k] = v
    return params

def objective_stage1(trial, symbol: str):
    try:
        feature_params_raw = suggest_feature_params(trial, yaml_cfg.get("features", {}))
        feature_cfg = FeatureConfig(**feature_params_raw)

        df = fetch_bars(symbol, cfg.timeframe, cfg.history_bars)
        X = build_features(df, feature_cfg, symbol)
        y = binary_up_down(df, cfg.prediction_horizon)
        data = merge_features_labels(df, X, y)

        X_train = data.drop(columns=["y", "close", "high", "low", "volume"])
        y_train = data["y"]

        model = MLStrategy(model="logreg", calibrate=False)
        tscv = TimeSeriesSplit(n_splits=5)
        aucs = []
        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            p_val = model.predict_proba(X_val)
            aucs.append(roc_auc_score(y_val, p_val))

        return 1 - float(pd.Series(aucs).mean())
    except Exception as e:
        logger.error(f"[Stage1] Error for {symbol}: {e}\n{traceback.format_exc()}")
        return float("inf")

# --- Stage 2: Full Feature + Model Tuning ---
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

def objective_stage2(trial, df, y, static_features, symbol, stage1_best_features):
    try:
        # --- Feature params ---
        feature_params_raw = stage1_best_features.copy()
        feature_params_raw.update(suggest_params(trial, "feature", yaml_cfg.get("features", {})))

        roc_lags_options = yaml_cfg.get("roc_lags_options", [
            [1, 3, 5, 10], [1, 2, 4, 8], [2, 5, 10, 15], [1, 2, 3]
        ])
        roc_lags_choice = trial.suggest_categorical("feature_roc_lags", roc_lags_options)
        feature_params_raw["roc_lags"] = roc_lags_choice

        feature_cfg = FeatureConfig(**feature_params_raw)
        X = build_dynamic_features(df, static_features, feature_cfg, symbol)
        data = merge_features_labels(df, X, y)

        X_train = data.drop(columns=["y", "close", "high", "low", "volume"])
        y_train = data["y"]

        # --- Model hyperparams ---
        model_params = {}
        for model in yaml_cfg["models"]:
            model_name = model["name"]
            model_params[model_name] = suggest_params(trial, f"model_{model_name}", model.get("params", {}))

        ens = Ensemble(cfg, model_params=model_params)

        # --- CV ---
        cv_samples_per_split = yaml_cfg.get("cv_samples_per_split", 300)
        n_splits_calculated = min(5, max(2, len(X_train) // cv_samples_per_split))
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

        return 1 - float(pd.Series(aucs).mean())
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"[Stage2] Error for {symbol}: {e}\n{traceback.format_exc()}")
        return float("inf")

# --- Runner per symbol ---
def run_hybrid_tuning(sym: str):
    logger.info(f"🔹 Hybrid tuning for {sym}")

    # Stage 1
    logger.info(f"[{sym}] Stage 1: Rapid feature tuning...")
    study1 = optuna.create_study(direction="minimize")
    study1.optimize(lambda trial: objective_stage1(trial, sym), n_trials=25)
    best_stage1 = {k.replace("feature_", ""): v for k, v in study1.best_params.items()}
    logger.info(f"[{sym}] Stage 1 best features: {best_stage1}")

    # Stage 2
    logger.info(f"[{sym}] Stage 2: Full ensemble tuning...")
    df = fetch_bars(sym, cfg.timeframe, cfg.history_bars)
    y = binary_up_down(df, cfg.prediction_horizon)
    static_features = build_static_features(df, symbol=sym)

    objective_partial = partial(objective_stage2, df=df, y=y, static_features=static_features,
                                symbol=sym, stage1_best_features=best_stage1)

    study_name = f"hybrid_tuning_{sym.replace('#','_')}_history_{cfg.history_bars}"
    storage_path = f"sqlite:///{os.path.join(PARAMS_DIR, study_name)}.db"
    pruner = optuna.pruners.MedianPruner()
    study2 = optuna.create_study(direction="minimize", study_name=study_name,
                                 storage=storage_path, load_if_exists=True, pruner=pruner)
    n_trials = yaml_cfg.get("optuna_n_trials", 100)
    study2.optimize(objective_partial, n_trials=n_trials)

    # --- Process results ---
    best_params_flat = study2.best_params
    best_params_structured = {"features": best_stage1.copy(), "models": {}}
    for key, value in best_params_flat.items():
        if key.startswith("feature_"):
            param_name = key.replace("feature_", "")
            best_params_structured["features"][param_name] = value
        elif key.startswith("model_"):
            parts = key.split("_")
            model_name = parts[1]
            param_name = "_".join(parts[2:])
            if model_name not in best_params_structured["models"]:
                best_params_structured["models"][model_name] = {}
            best_params_structured["models"][model_name][param_name] = value

    # Save pickle (for train_adaptive.py)
    pkl_path = os.path.join(PARAMS_DIR, f"{sym.replace('#','_')}_hybrid_best.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(best_params_structured, f)
    logger.info(f"[{sym}] Hybrid best params saved → {pkl_path}")

    # Save YAML (for readability)
    yaml_path = pkl_path.replace(".pkl", ".yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(best_params_structured, f, default_flow_style=False)
    logger.info(f"[{sym}] Hybrid best params saved (YAML) → {yaml_path}")

    # Save plots
    try:
        fig = optuna.visualization.plot_optimization_history(study2)
        fig.write_image(pkl_path.replace(".pkl", "_history.png"))
        fig = optuna.visualization.plot_param_importances(study2)
        fig.write_image(pkl_path.replace(".pkl", "_importances.png"))
    except Exception as e:
        logger.warning(f"[{sym}] Could not save plots: {e}")

# --- Main ---
if __name__ == "__main__":
    n_jobs = yaml_cfg.get("n_jobs", -1)
    Parallel(n_jobs=n_jobs)(delayed(run_hybrid_tuning)(sym) for sym in cfg.symbols)
