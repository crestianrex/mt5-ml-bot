# src/utils.py
from __future__ import annotations
import os
import pickle
import sys
import copy
from loguru import logger
import pandas as pd
from src.features import FeatureConfig, build_static_features, build_dynamic_features, add_contextual_features
from src.labels import binary_up_down
from src.ensemble import Ensemble
from src import data_manager

MODEL_DIR = "models"
PARAMS_DIR = "optuna_params"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

def setup_logging(level="INFO", to_file=True, rotate="10 MB", retention="7 days"):
    logger.remove()
    # Use enqueue=True to make logging from multiple processes safe for shared sinks (stderr and the log file).
    # This will prevent messages from being garbled, but they will still be mixed.
    # The log messages themselves should contain context like the symbol name.
    logger.add(sys.stderr, level=level, enqueue=True)
    if to_file:
        os.makedirs("logs", exist_ok=True)
        logger.add("logs/bot.log", level=level, rotation=rotate, retention=retention, enqueue=True)

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

def get_training_data(cfg: Cfg, symbol: str, feature_cfg: FeatureConfig, count: int | None = None, source: str = "csv", load_all_data: bool = False, build_dynamic: bool = True):
    """
    New centralized data pipeline.
    - If build_dynamic is True, returns (data, X, y) for trainers/backtesters.
    - If build_dynamic is False, returns (X, y, df) for the tuner.
    """
    fetch_count = None if load_all_data else (count if count is not None else cfg.history_bars)

    # --- 1. Initialize DataManager ---
    dm = data_manager.DataManager(cfg)

    # --- 2. Fetch All Dataframes ---
    logger.info(f"[{symbol}] Fetching primary data ({fetch_count or 'all'} bars, {cfg.timeframe}) from {cfg.data_source.upper()}...")
    if cfg.data_source == "csv":
        df = dm.load_local_history(symbol, cfg.timeframe, count=fetch_count)
    elif cfg.data_source == "mt5":
        df = dm._fetch_bars_from_mt5_chunked(symbol, cfg.timeframe, fetch_count)
    else:
        raise ValueError(f"Unknown data source: {cfg.data_source}")

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64")

    mta_df = None
    if cfg.context_features.mta.enabled:
        logger.info(f"[{symbol}] Fetching MTA data ({fetch_count or 'all'} bars, {cfg.context_features.mta.timeframe}) from {cfg.data_source.upper()}...")
        if cfg.data_source == "csv":
            mta_df = dm.load_local_history(symbol, cfg.context_features.mta.timeframe, count=fetch_count)
        elif cfg.data_source == "mt5":
            mta_df = dm._fetch_bars_from_mt5_chunked(symbol, cfg.context_features.mta.timeframe, fetch_count)
        else:
            raise ValueError(f"Unknown data source: {cfg.data_source}")
        if mta_df.empty:
            logger.warning(f"[{symbol}] No MTA data fetched for timeframe {cfg.context_features.mta.timeframe}.")
            mta_df = None

    inter_market_df = None
    if cfg.context_features.inter_market.enabled:
        im_sym = cfg.context_features.inter_market.symbol
        logger.info(f"[{symbol}] Fetching Inter-Market data for {im_sym} ({fetch_count or 'all'} bars, {cfg.timeframe}) from {cfg.data_source.upper()}...")
        if cfg.data_source == "csv":
            inter_market_df = dm.load_local_history(im_sym, cfg.timeframe, count=fetch_count)
        elif cfg.data_source == "mt5":
            inter_market_df = dm._fetch_bars_from_mt5_chunked(im_sym, cfg.timeframe, fetch_count)
        else:
            raise ValueError(f"Unknown data source: {cfg.data_source}")
        if inter_market_df.empty:
            logger.warning(f"[{symbol}] No Inter-Market data fetched for symbol {im_sym}.")
            inter_market_df = None

    # --- 3. Build Feature Set ---
    logger.info(f"[{symbol}] Building full feature set...")
    
    # Build all features using the unified build_features function
    X = build_features(df.copy(), feature_cfg, cfg, symbol=symbol, mta_df=mta_df, inter_market_df=inter_market_df)
    y = binary_up_down(df, cfg.prediction_horizon)

    # Align X and y by index
    aligned_idx = X.index.intersection(y.index)
    X = X.loc[aligned_idx]
    y = y.loc[aligned_idx]

    if not build_dynamic:
        # Return the intermediate artifacts needed by the tuner
        logger.info(f"[{symbol}] Data pipeline complete for tuner. Returning features and labels.")
        return X, y, df # Return X, y, df for consistency

    # For trainer/backtester, X and y are already built
    data = data_manager.merge_features_labels(df, X, y)

    if data is None or data.empty:
        return pd.DataFrame(), X if X is not None else pd.DataFrame(), y if y is not None else pd.Series(dtype="float64")
    
    logger.info(f"[{symbol}] Data pipeline complete. Final shape: {data.shape}")
    return data, X, y

def load_ensemble(cfg: Cfg, symbol: str) -> Ensemble:
    # New: ensemble is saved in a directory, not a single file
    model_dir_path = os.path.join(MODEL_DIR, f"{symbol.replace('#','')}_ensemble")
    model_params = load_optuna_params(symbol)

    if os.path.isdir(model_dir_path):
        logger.info(f"[{symbol}] Loading saved ensemble from directory {model_dir_path}")
        try:
            # Use the new class method to load
            return Ensemble.load(model_dir_path, cfg, model_params=model_params)
        except Exception as e:
            logger.exception(f"[{symbol}] Failed to load ensemble from directory: {e}; creating a new one.")

    logger.info(f"[{symbol}] No saved ensemble directory found. Creating a new one.")
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

def safe_retrain_ensemble(cfg: Cfg, symbol: str, ens_old: Ensemble, X_train: pd.DataFrame, y_train: pd.Series, prices: pd.Series, dry_run: bool = False) -> Ensemble:
    """
    Safely retrains an ensemble model.

    Args:
        cfg: The configuration object.
        symbol: The symbol being trained.
        ens_old: The existing ensemble model.
        X_train: The training features.
        y_train: The training labels.
        prices: The close prices for the training period.
        dry_run: If True, the new model will not be saved.

    Returns:
        The retrained ensemble if it's better than the old one, otherwise the old ensemble.
    """
    logger.info(f"[{symbol}] Starting safe retraining...")
    
    ens_new = copy.deepcopy(ens_old)

    try:
        ens_new.fit(X_train, y_train, prices=prices)
        new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))
        old_auc = getattr(ens_old, "ensemble_cv_auc_", getattr(ens_old, "cv_auc_", None))

        if new_auc is None:
            logger.warning(f"[{symbol}] New ensemble reports no AUC; refusing to replace.")
            return ens_old

        if old_auc is None or (new_auc - old_auc) >= cfg.risk.min_auc_improvement:
            if not dry_run:
                save_ensemble(ens_new, symbol)
            logger.info(f"[{symbol}] Retrain accepted. old_auc={old_auc} new_auc={new_auc}")
            return ens_new
        else:
            logger.info(f"[{symbol}] Retrain NOT accepted. improvement {(new_auc - old_auc):.4f} < {cfg.risk.min_auc_improvement}")
            return ens_old
    except Exception as e:
        logger.exception(f"[{symbol}] Retraining failed: {e}")
        return ens_old
