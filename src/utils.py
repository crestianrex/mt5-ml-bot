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
    """Sets up the Loguru logger."""
    logger.remove()
    logger.add(sys.stderr, level=level)
    if to_file:
        logger.add("logs/bot.log", level=level, rotation=rotate, retention=retention)

def load_optuna_params(symbol: str) -> dict | None:
    """Load per-symbol Optuna-tuned hyperparameters if available."""
    file_path = os.path.join(PARAMS_DIR, f"{symbol.replace('#','')}_best_params.pkl")
    if not os.path.exists(file_path):
        logger.warning(f"[{symbol}] No Optuna params found at {file_path}, using defaults from config.")
        return None
    
    with open(file_path, "rb") as f:
        best_params_flat = pickle.load(f)
        
    model_params = {"lgbm": {}, "xgb": {}, "rf": {}, "logreg": {}}
    for k, v in best_params_flat.items():
        for model_name in model_params:
            if k.startswith(model_name):
                param_name = k[len(model_name)+1:]
                model_params[model_name][param_name] = v
                
    logger.info(f"[{symbol}] Loaded Optuna best params from {file_path}")
    return model_params

def get_training_data(cfg: Cfg, symbol: str, source: str = "mt5") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fetches historical data and builds features and labels."""
    
    if source == "mt5":
        from src.data import fetch_bars, merge_features_labels
        logger.info(f"[{symbol}] Fetching {cfg.history_bars} bars for timeframe {cfg.timeframe} from MT5...")
    elif source == "csv":
        from src.data_colab import fetch_bars, merge_features_labels
        logger.info(f"[{symbol}] Fetching {cfg.history_bars} bars for timeframe {cfg.timeframe} from CSV...")
    else:
        raise ValueError(f"Unknown data source: {source}")

    df = fetch_bars(symbol, cfg.timeframe, cfg.history_bars)
    if df.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
        
    logger.info(f"[{symbol}] Building features...")
    feature_cfg = FeatureCfg(**cfg.features.__dict__)
    X = build_features(df, feature_cfg, symbol=symbol, timeframe_minutes=cfg.timeframe)
    
    logger.info(f"[{symbol}] Building labels...")
    y = binary_up_down(df, cfg.prediction_horizon)
    
    data = merge_features_labels(df, X, y)
    return data, X, y

def load_ensemble(cfg: Cfg, symbol: str) -> Ensemble:
    """Loads a saved ensemble model or creates a new one."""
    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('#','')}_ensemble.pkl")
    if os.path.exists(model_path):
        logger.info(f"[{symbol}] Loading saved ensemble from {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    logger.info(f"[{symbol}] No saved ensemble found. Creating a new one.")
    model_params = load_optuna_params(symbol)
    ens = Ensemble(cfg, model_params=model_params)
    return ens

def save_ensemble(ensemble: Ensemble, symbol: str):
    """Saves the ensemble model to a file."""
    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('#','')}_ensemble.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"[{symbol}] Ensemble model saved to {model_path}")