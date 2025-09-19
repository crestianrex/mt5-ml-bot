# src/utils.py
from __future__ import annotations
import os
import pickle
import sys
from loguru import logger
import pandas as pd
from src.config import Cfg, FeatureCfg
from src.features import build_features
from src.labels import binary_up_down
from src.ensemble import Ensemble

MODEL_DIR = "models"
PARAMS_DIR = "optuna_params"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)


def setup_logging(level="INFO", to_file=True, rotate="10 MB", retention="7 days"):
    logger.remove()
    logger.add(sys.stderr, level=level)
    if to_file:
        os.makedirs("logs", exist_ok=True)
        logger.add("logs/bot.log", level=level, rotation=rotate, retention=retention)


def load_optuna_params(symbol: str) -> dict | None:
    file_path = os.path.join(PARAMS_DIR, f"{symbol.replace('#','')}_best_params.pkl")
    if not os.path.exists(file_path):
        logger.warning(f"[{symbol}] No Optuna params found at {file_path}, using defaults from config.")
        return None
    try:
        with open(file_path, "rb") as f:
            loaded_params = pickle.load(f)
    except Exception as e:
        logger.error(f"[{symbol}] Failed to load optuna params: {e}")
        return None

    if "models" not in loaded_params:
        logger.warning(f"[{symbol}] 'models' key not found in Optuna params; using empty model params.")
        return {"lgbm": {}, "xgb": {}, "rf": {}, "logreg": {}}
    model_params = loaded_params["models"]
    logger.info(f"[{symbol}] Loaded Optuna best params from {file_path}")
    return model_params


def get_training_data(cfg: Cfg, symbol: str, count: int | None = None, source: str = "csv", load_all_data: bool = False):
    """
    Returns (data, X, y):
      - data: merged DataFrame with features+labels
      - X: features DataFrame
      - y: labels Series
    On error or if insufficient data, returns (empty_df, empty_df, empty_series)
    """
    if load_all_data:
        fetch_count = None
        logger.info(f"[{symbol}] Fetching ALL available bars for timeframe {cfg.timeframe} from {source.upper()}...")
    else:
        fetch_count = count if count is not None else cfg.history_bars
        logger.info(f"[{symbol}] Fetching {fetch_count} bars for timeframe {cfg.timeframe} from {source.upper()}...")

    if source == "mt5":
        from src.data import fetch_bars, merge_features_labels
    elif source == "csv":
        from src.data_colab import fetch_bars, merge_features_labels
    else:
        raise ValueError(f"Unknown data source: {source}")

    df = fetch_bars(symbol, cfg.timeframe, fetch_count)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64")

    logger.info(f"[{symbol}] Building features...")
    feature_cfg = FeatureCfg(**(cfg.features.__dict__ if hasattr(cfg, "features") else {}))
    timeframe_minutes = (cfg.timeframe_seconds() // 60) if cfg.timeframe_seconds() else None
    X = build_features(df, feature_cfg, symbol=symbol, timeframe_minutes=timeframe_minutes)
    logger.info(f"[{symbol}] Building labels...")
    y = binary_up_down(df, cfg.prediction_horizon)
    data = merge_features_labels(df, X, y)
    if data is None or data.empty:
        return pd.DataFrame(), X if X is not None else pd.DataFrame(), y if y is not None else pd.Series(dtype="float64")
    return data, X, y


def load_ensemble(cfg: Cfg, symbol: str) -> Ensemble:
    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('#','')}_ensemble.pkl")
    if os.path.exists(model_path):
        logger.info(f"[{symbol}] Loading saved ensemble from {model_path}")
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to load ensemble: {e}; creating a new one.")
    logger.info(f"[{symbol}] No saved ensemble found. Creating a new one.")
    model_params = load_optuna_params(symbol)
    ens = Ensemble(cfg, model_params=model_params)
    return ens


def save_ensemble(ensemble: Ensemble, symbol: str):
    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('#','')}_ensemble.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"[{symbol}] Ensemble model saved to {model_path}")
