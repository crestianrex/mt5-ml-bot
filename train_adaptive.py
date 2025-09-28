# train_adaptive.py
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
from src.utils import get_training_data, load_ensemble, save_ensemble
from src.ensemble import Ensemble

# Safe retraining parameters
MIN_SAMPLES_TO_RETRAIN = 400  # don't retrain if less than this many samples
MIN_AUC_IMPROVEMENT = 0.005  # require this improvement to accept new model

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

    # fetch all available training data
    data, X, y = get_training_data(cfg, symbol, load_all_data=True, source=cfg.data_source if hasattr(cfg, "data_source") else "mt5")
    if X is None or X.empty or len(X) < MIN_SAMPLES_TO_RETRAIN:
        msg = f"[{symbol}] Not enough data to retrain: {0 if X is None else len(X)} samples"
        logger.warning(msg)
        return {"ok": False, "reason": msg}

    # backup current ensemble
    ens_backup = copy.deepcopy(ens_old)

    try:
        ens_new = copy.deepcopy(ens_old)
        # Fit on all historical data
        ens_new.fit(X, y, prices=data["close"] if "close" in data.columns else None)
        new_auc = getattr(ens_new, "ensemble_cv_auc_", getattr(ens_new, "cv_auc_", None))
        old_auc = getattr(ens_old, "ensemble_cv_auc_", getattr(ens_old, "cv_auc_", None))
        logger.info(f"[{symbol}] old_auc={old_auc} new_auc={new_auc}")

        if new_auc is None:
            logger.warning(f"[{symbol}] New ensemble reports no AUC; refusing to replace.")
            return {"ok": False, "reason": "no_new_auc"}

        if old_auc is None:
            accept = True
        else:
            # accept only if improved by threshold
            accept = (new_auc - old_auc) >= MIN_AUC_IMPROVEMENT

        if accept:
            if not dry_run:
                save_ensemble(ens_new, symbol)
            logger.info(f"[{symbol}] Retrain accepted. old_auc={old_auc} new_auc={new_auc}")
            return {"ok": True, "old_auc": old_auc, "new_auc": new_auc}
        else:
            logger.info(f"[{symbol}] Retrain NOT accepted. improvement {(new_auc-old_auc):.4f} < {MIN_AUC_IMPROVEMENT}")
            return {"ok": False, "reason": "insufficient_improvement", "old_auc": old_auc, "new_auc": new_auc}

    except Exception as e:
        logger.exception(f"[{symbol}] Retrain failed: {e}")
        # restore backup if needed
        try:
            save_ensemble(ens_backup, symbol)
        except Exception:
            pass
        return {"ok": False, "reason": "exception", "err": str(e)}


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
