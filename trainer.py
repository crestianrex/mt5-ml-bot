# trainer.py
"""
Adaptive retraining module. Intended to be run periodically or from the main loop.
Performs safe retraining with no lookahead. On failure, keeps the previous ensemble.
Saves ensembles via utils.save_ensemble() and optionally writes training artifacts.
"""

from __future__ import annotations
import os
import copy
import pickle
from loguru import logger
from datetime import datetime, timezone
from typing import Optional

from src.config import Cfg
from src.features import FeatureConfig
from src.utils import get_training_data, load_ensemble, save_ensemble, safe_retrain_ensemble, load_optuna_params
from src.ensemble import Ensemble

# Safe retraining parameters
MIN_SAMPLES_TO_RETRAIN = 400  # don't retrain if less than this many samples

def retrain_symbol(cfg: Cfg, symbol: str, dry_run: bool = True) -> dict:
    """
    Retrain ensemble for one symbol safely:
     - load full historical data (safe: only past bars)
     - backup current ensemble
     - fit new ensemble on full past data
     - evaluate with time-series CV (internal to ensemble.fit)
     - accept new model only if ensemble_cv_auc_ improved by MIN_AUC_IMPROVEMENT
    Returns a dict with status and metadata.
    """
    logger.info(f"[{symbol}] Starting safe retrain (dry_run={dry_run})")
    ens_old = load_ensemble(cfg, symbol)
    if ens_old is None:
        logger.info(f"[{symbol}] No existing ensemble; creating a new one")
        ens_old = Ensemble(cfg)

    # Load best feature params from optuna study
    optuna_params = load_optuna_params(symbol)
    feature_params = optuna_params.get('features', {})
    feature_cfg = FeatureConfig(**feature_params)

    # fetch all available training data using the new centralized pipeline
    data, X, y = get_training_data(
        cfg, 
        symbol, 
        feature_cfg=feature_cfg, 
        load_all_data=True, 
        source=cfg.data_source if hasattr(cfg, "data_source") else "mt5"
    )

    if X is None or X.empty or len(X) < MIN_SAMPLES_TO_RETRAIN:
        msg = f"[{symbol}] Not enough data to retrain: {0 if X is None else len(X)} samples"
        logger.warning(msg)
        return {"ok": False, "reason": msg}

    # Use the shared safe_retrain_ensemble function
    ens_new = safe_retrain_ensemble(cfg, symbol, ens_old, X, y, data["close"] if "close" in data.columns else None, dry_run=dry_run)

    # Check if the ensemble was updated (i.e., if ens_new is different from ens_old)
    if ens_new is ens_old:
        # Retrain was not accepted or failed
        new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))
        old_auc = getattr(ens_old, "ensemble_cv_auc_", getattr(ens_old, "cv_auc_", None))
        return {"ok": False, "reason": "insufficient_improvement_or_failed", "old_auc": old_auc, "new_auc": new_auc}
    else:
        # Retrain was accepted
        new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))
        old_auc = getattr(ens_old, "ensemble_cv_auc_", getattr(ens_old, "cv_auc_", None))
        return {"ok": True, "old_auc": old_auc, "new_auc": new_auc}


def retrain_all(cfg: Cfg, symbols: list[str], dry_run: bool = True) -> dict:
    results = {}
    for s in symbols:
        try:
            results[s] = retrain_symbol(cfg, s, dry_run=dry_run)
        except Exception as e:
            logger.exception(f"[{s}] retrain_all error: {e}")
            results[s] = {"ok": False, "reason": str(e)}
    return results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    try:
        import MetaTrader5 as mt5 # type: ignore
    except ImportError:
        mt5 = None
        logger.warning("MetaTrader5 module not found. Live MT5 operations will be disabled.")
    from src.config import Cfg

    load_dotenv()

    # Establish MT5 connection
    if mt5:
        login_id_str = os.getenv("MT5_LOGIN")
        if not login_id_str:
            print("MT5_LOGIN not found in environment variables. Exiting.")
            quit()

        if not mt5.initialize(
            login=int(login_id_str),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path=os.getenv("MT5_PATH")
        ):
            print(f"mt5.initialize() failed, error code = {mt5.last_error()}")
            quit()
        
        print("MT5 connection initialized.")
    else:
        print("MT5 connection skipped: MetaTrader5 module not available.")
    
    try:
        cfg = Cfg.from_yaml("config.yaml")
        symbols = getattr(cfg, "symbols", [])
        if not symbols:
            print("No symbols found in config.yaml. Exiting.")
            quit()
        print("Running retrain_all with dry_run=False (model files will be overwritten).")
        res = retrain_all(cfg, symbols, dry_run=False)
        print(res)
    finally:
        # Shutdown MT5 connection
        if mt5:
            mt5.shutdown()
            print("MT5 connection shut down.")
