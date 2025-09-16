# main_hybrid_adaptive.py
import os
import time
import copy
from dotenv import load_dotenv
from loguru import logger
import pandas as pd

from src.config import Cfg
from src.mt5_client import MT5Client
from src.risk import RiskManager
from src.execution import Execution
from src.utils import setup_logging, get_training_data, load_ensemble, save_ensemble

# --- Initial Setup ---
load_dotenv()
setup_logging()

def print_dashboard(cfg, risk, ens_per_symbol, X_per_symbol, bar_counter):
    """
    Prints a live portfolio dashboard per symbol, throttled every N bars.
    """
    if bar_counter % cfg.dashboard_every_bars != 0:
        return  # throttle dashboard logging

    import MetaTrader5 as mt5
    account_info = mt5.account_info()
    equity = account_info.equity if account_info else 0
    balance = account_info.balance if account_info else 0
    drawdown = 1 - equity / balance if balance else 0
    total_open_risk = sum([pos.risk for pos in risk.open_positions_cache.values()])

    logger.info("=== PORTFOLIO DASHBOARD ===")
    logger.info(f"Equity: {equity:.2f} | Balance: {balance:.2f} | Drawdown: {drawdown:.3%} | Total Open Risk: {total_open_risk:.3%}")
    for sym in cfg.symbols:
        X = X_per_symbol.get(sym)
        if X is not None and not X.empty:
            atr = X["atr_14"].iloc[-1]
            last_features = X.iloc[[-1]]
            ens = ens_per_symbol[sym]
            prob_up = ens.predict_proba(last_features).iloc[0]
            open_pos = risk.open_positions_cache.get(sym, 'None')
            logger.info(f"[{sym}] ATR={atr:.5f} | p_up={prob_up:.3f} | Open Positions: {open_pos}")

def run(dry_run: bool = False):
    """
    Production-ready main loop for hybrid adaptive MT5 ML bot.

    Key Features:
    - risk.should_trade() check before attempting trades
    - Safe incremental retraining with AUC validation
    - Portfolio-level risk tracking
    - Live dashboard throttled per N bars
    - Dry-run mode for testing without live trades
    """
    cfg = Cfg.from_yaml("config.yaml")
    cfg.dashboard_every_bars = getattr(cfg, "dashboard_every_bars", 1)  # default to every bar

    logger.info("=== Starting MT5 ML Bot (Hybrid Adaptive) ===")
    logger.info(f"Dry-run mode: {dry_run}")
    logger.info(f"Symbols: {cfg.symbols}")
    logger.info(f"Trading session: {cfg.risk.session_filter['start']} - {cfg.risk.session_filter['end']}")
    logger.info(f"Max portfolio risk: {cfg.risk.max_portfolio_risk:.2%}")
    logger.info(f"ATR SL multiplier: {cfg.risk.atr_multiplier_sl}, Trailing ATR: {cfg.risk.trailing_atr_mult}")
    logger.info(f"Dynamic risk enabled: {cfg.risk.dynamic_risk['enabled']}, Dynamic TP enabled: {cfg.risk.dynamic_tp['enabled']}")
    logger.info(f"Ensemble method: {cfg.ensemble.method}, Threshold metric: {cfg.ensemble.threshold_metric}")

    # --- MT5 Connection ---
    mt5c = MT5Client(
        os.getenv("MT5_LOGIN"),
        os.getenv("MT5_PASSWORD"),
        os.getenv("MT5_SERVER"),
        os.getenv("MT5_PATH")
    )
    if not mt5c.connect():
        logger.error("MT5 connection failed. Exiting.")
        return

    # --- Load Ensembles ---
    ens_per_symbol = {sym: load_ensemble(cfg, sym) for sym in cfg.symbols}
    bar_counters = {sym: 0 for sym in cfg.symbols}
    last_bar_time = {sym: None for sym in cfg.symbols}
    X_per_symbol = {}

    risk = RiskManager(cfg.risk)

    try:
        while True:
            for sym in cfg.symbols:
                try:
                    # --- Fetch latest bar data (minimal history for speed) ---
                    temp_cfg = Cfg.from_yaml("config.yaml")
                    temp_cfg.history_bars = 500
                    data, X, y = get_training_data(temp_cfg, sym)
                    if data.empty:
                        continue

                    X_per_symbol[sym] = X
                    latest_bar_time = data.index[-1]

                    if last_bar_time[sym] == latest_bar_time:
                        continue  # skip if no new bar
                    last_bar_time[sym] = latest_bar_time
                    bar_counters[sym] += 1
                    logger.info(f"[{sym}] New bar detected at {latest_bar_time}")

                    # --- Safe Incremental Retraining ---
                    if bar_counters[sym] % cfg.retrain_every_bars == 0:
                        logger.info(f"[{sym}] Starting safe retraining...")
                        full_data, full_X, full_y = get_training_data(cfg, sym)
                        ens_copy = copy.deepcopy(ens_per_symbol[sym])
                        ens_copy.fit(full_X, full_y)
                        auc_new = getattr(ens_copy, "ensemble_cv_auc_", 0)
                        if auc_new >= cfg.risk.min_ensemble_auc:
                            ens_per_symbol[sym] = ens_copy
                            save_ensemble(ens_copy, sym)
                            logger.info(f"[{sym}] Retraining successful, AUC={auc_new:.3f}")
                        else:
                            logger.warning(f"[{sym}] Retraining skipped: AUC={auc_new:.3f} below min {cfg.risk.min_ensemble_auc}")

                    # --- Manage existing trades first ---
                    atr = X["atr_14"].iloc[-1]
                    last_features = X.iloc[[-1]]
                    exe = Execution(ens_per_symbol[sym], risk, dry_run=dry_run)
                    exe.manage_trades(sym, atr)

                    # --- Update portfolio-level open risk AFTER managing trades ---
                    total_open_risk = sum([pos.risk for pos in risk.open_positions_cache.values()])

                    # --- Check trading permission ---
                    import MetaTrader5 as mt5
                    account_info = mt5.account_info()
                    equity = account_info.equity if account_info else 0
                    balance = account_info.balance if account_info else 0
                    drawdown = 1 - equity / balance if balance else 0

                    if not risk.should_trade(pd.Timestamp.now(), drawdown):
                        logger.info(f"[{sym}] Trade skipped due to drawdown/session rules")
                        continue

                    # --- Execute trade ---
                    result = exe.trade(sym, last_features, atr, ens_per_symbol[sym].ensemble_cv_auc_, total_open_risk)
                    if result.ok:
                        logger.info(f"[{sym}] Trade executed: {result.message}")
                    else:
                        logger.info(f"[{sym}] Trade skipped: {result.message}")

                except Exception as e:
                    logger.exception(f"[{sym}] Loop error: {e}")

            # --- Live Dashboard (throttled) ---
            print_dashboard(cfg, risk, ens_per_symbol, X_per_symbol, bar_counter=sum(bar_counters.values()))

            time.sleep(cfg.timeframe_seconds() or 60)

    except KeyboardInterrupt:
        logger.info("=== Stopping MT5 ML Bot ===")
    finally:
        mt5c.shutdown()
        logger.info("MT5 shutdown complete.")


if __name__ == "__main__":
    run(dry_run=False)  # Change to False for live trading
