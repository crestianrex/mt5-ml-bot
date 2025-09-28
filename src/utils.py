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
    # symbol names in params are saved without '#'
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

    if not isinstance(loaded_params, dict) or "models" not in loaded_params:
        logger.warning(f"[{symbol}] Optuna params format unexpected; using empty model params.")
        return {"lgbm": {}, "xgb": {}, "rf": {}, "logreg": {}}

    logger.info(f"[{symbol}] Loaded Optuna best params from {file_path}")
    return loaded_params.get("models", {})

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
    else:
        fetch_count = count if count is not None else cfg.history_bars

    # If on a non-Windows platform, MT5 is not available, so force CSV usage.
    if sys.platform != "win32" and source == "mt5":
        logger.warning(f"MT5 is not supported on {sys.platform}. Switching to 'csv' data source.")
        source = "csv"
        
    if load_all_data:
        logger.info(f"[{symbol}] Fetching ALL available bars for timeframe {cfg.timeframe} from {source.upper()}...")
    else:
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
    # New: ensemble is saved in a directory, not a single file
    model_dir_path = os.path.join(MODEL_DIR, f"{symbol.replace('#','')}_ensemble")
    if os.path.isdir(model_dir_path):
        logger.info(f"[{symbol}] Loading saved ensemble from directory {model_dir_path}")
        try:
            # Use the new class method to load
            return Ensemble.load(model_dir_path, cfg)
        except Exception as e:
            logger.exception(f"[{symbol}] Failed to load ensemble from directory: {e}; creating a new one.")

    logger.info(f"[{symbol}] No saved ensemble directory found. Creating a new one.")
    model_params = load_optuna_params(symbol)
    ens = Ensemble(cfg, model_params=model_params)
    return ens

def save_ensemble(ensemble: Ensemble, symbol: str):
    # New: save to a directory
    model_dir_path = os.path.join(MODEL_DIR, f"{symbol.replace('#','')}_ensemble")
    try:
        # Use the new instance method to save
        ensemble.save(model_dir_path)
        logger.info(f"[{symbol}] Ensemble model saved to directory {model_dir_path}")
    except Exception as e:
        logger.error(f"[{symbol}] Failed to save ensemble: {e}")
