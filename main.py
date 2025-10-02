# main.py
import os
import time
import copy
from multiprocessing import Process
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
from src.config import Cfg
from src.features import FeatureConfig
import MetaTrader5 as mt5  # type: ignore
from src.mt5_client import MT5Client
from src.risk import RiskManager
from src.execution import Execution
from src.utils import setup_logging, get_training_data, load_ensemble, save_ensemble, safe_retrain_ensemble, load_optuna_params
from src.live_performance_monitor import LivePerformanceMonitor
from src.notifier import TelegramNotifier # NEW

# --- Initial Setup ---
load_dotenv()
setup_logging()

def print_dashboard(cfg, risk, ens_per_symbol, X_per_symbol, bar_counter):
    """ Prints a live portfolio dashboard per symbol, throttled every N bars. """
    if bar_counter % cfg.dashboard_every_bars != 0:
        return  # throttle dashboard logging

    account_info = mt5.account_info()
    equity = getattr(account_info, "equity", 0.0) if account_info else 0.0
    balance = getattr(account_info, "balance", 0.0) if account_info else 0.0
    drawdown = 0.0
    try:
        drawdown = 1 - (equity / balance) if balance else 0.0
    except Exception:
        drawdown = 0.0

    total_open_risk = sum([pos.get('risk', 0.0) for pos in risk.open_positions_cache.values()])

    logger.info("=== PORTFOLIO DASHBOARD ===")
    logger.info(f"Equity: {equity:.2f} | Balance: {balance:.2f} | Drawdown: {drawdown:.3%} | Total Open Risk: {total_open_risk:.3%}")

    for sym in cfg.symbols:
        X = X_per_symbol.get(sym)
        ens = ens_per_symbol.get(sym)
        if X is not None and not X.empty and ens is not None:
            try:
                atr = float(X["atr_14"].iloc[-1])
            except Exception:
                atr = 0.0
            try:
                last_features = X.iloc[[-1]]
                prob_up = ens.predict_proba(last_features)
                # normalize to scalar
                if hasattr(prob_up, "iloc"):
                    p_up = float(prob_up.iloc[0])
                elif isinstance(prob_up, (list, tuple,)):
                    p_up = float(prob_up[0])
                else:
                    p_up = float(prob_up)
            except Exception:
                p_up = 0.5
            open_pos = risk.open_positions_cache.get(sym, 'None')
            logger.info(f"[{sym}] ATR={atr:.5f} | p_up={p_up:.3f} | Open Positions: {open_pos}")

def run_retraining_in_background(cfg, sym, feature_cfg, dry_run, notifier):
    """
    A wrapper function to run the entire retraining pipeline in a separate process.
    """
    try:
        logger.info(f"[{sym}] Background process started for retraining.")
        full_data, full_X, full_y = get_training_data(cfg, sym, feature_cfg=feature_cfg, source=cfg.data_source, load_all_data=True)
        
        if full_X.empty or full_y.empty:
            message = f"[{sym}] <b>WARNING:</b> No data for retraining, background process exiting."
            logger.warning(message)
            if notifier: notifier.send_message(message, level="WARNING")
            return

        ens_old = load_ensemble(cfg, sym)
        
        safe_retrain_ensemble(cfg, sym, ens_old, full_X, full_y, full_data["close"] if "close" in full_data.columns else None, dry_run=dry_run)
        logger.info(f"[{sym}] Background retraining process finished.")
    except Exception as e:
        logger.exception(f"[{sym}] Background retraining process failed: {e}")


def run(dry_run: bool = False):
    """ Production-ready main loop for hybrid adaptive MT5 ML bot. """
    cfg = Cfg.from_yaml("config.yaml")
    cfg.dashboard_every_bars = getattr(cfg, "dashboard_every_bars", 1)

    logger.info("=== Starting MT5 ML Bot (Hybrid Adaptive) ===")
    logger.info(f"Dry-run mode: {dry_run}")
    logger.info(f"Symbols: {cfg.symbols if hasattr(cfg,'symbols') else []}")

    # --- MT5 Connection ---
    mt5c = MT5Client(
        os.getenv("MT5_LOGIN"),
        os.getenv("MT5_PASSWORD"),
        os.getenv("MT5_SERVER"),
        os.getenv("MT5_PATH"),
    )
    if not mt5c.connect():
        logger.error("MT5 connection failed. Exiting.")
        notifier.send_message("<b>CRITICAL:</b> MT5 connection failed. Bot exiting.", level="CRITICAL")
        return

    # Get initial equity from MT5 account info
    account_info = mt5.account_info()
    initial_equity = getattr(account_info, "equity", 100.0) if account_info else 100.0
    cfg.initial_equity = initial_equity # Set initial equity in Cfg for the monitor

    live_monitor = LivePerformanceMonitor(cfg)
    live_monitor.load_state() # NEW: Load previous state on startup

    notifier = TelegramNotifier(cfg) # NEW: Initialize notifier

    # --- Load Ensembles and Feature Configs ---
    ens_per_symbol = {}
    active_model_auc = {} # NEW: To store AUC of currently active model
    for sym in cfg.symbols:
        ens_per_symbol[sym] = load_ensemble(cfg, sym)
        active_model_auc[sym] = getattr(ens_per_symbol[sym], "ensemble_cv_auc_", getattr(ens_per_symbol[sym], "cv_auc_", 0.5)) # Initialize with loaded model's AUC

        optuna_params = load_optuna_params(sym)
        feature_params = optuna_params.get('features', {}) if optuna_params else {}

        feature_cfg_per_symbol[sym] = FeatureConfig(**feature_params)

    bar_counters = {sym: 0 for sym in cfg.symbols}
    last_bar_time = {sym: None for sym in cfg.symbols}
    X_per_symbol = {}
    risk = RiskManager(cfg, notifier=notifier) # NEW: Pass notifier
    retraining_processes = {}
    retraining_status = {sym: False for sym in cfg.symbols} # NEW: Track if retraining is active
    trading_blocked_by_low_new_model_auc = {sym: False for sym in cfg.symbols} # NEW: Track if trading is blocked due to low new model AUC

    try:
        while True:
            # refresh account info once per loop
            account_info = mt5.account_info()
            equity = getattr(account_info, "equity", 0.0) if account_info else 0.0
            live_monitor.update_equity(datetime.datetime.now(datetime.timezone.utc), equity)
            balance = getattr(account_info, "balance", 0.0) if account_info else 0.0
            drawdown = 0.0
            try:
                drawdown = 1 - (equity / balance) if balance else 0.0
            except Exception:
                drawdown = 0.0

            for sym in cfg.symbols:
                try:
                    # --- Check for finished retraining processes ---
                    if sym in retraining_processes and not retraining_processes[sym].is_alive():
                        logger.info(f"[{sym}] Background retraining process finished. Joining process and reloading model.")
                        retraining_processes[sym].join()  # Clean up the finished process
                        
                        # NEW: Selective acceptance of retrained model with absolute floor check
                        new_ens = load_ensemble(cfg, sym) # Load the newly retrained model temporarily
                        new_model_auc = getattr(new_ens, "ensemble_cv_auc_", getattr(new_ens, "cv_auc_", 0.5))
                        current_active_auc = active_model_auc[sym]
                        min_auc_improvement = cfg.risk.min_auc_improvement
                        min_ensemble_auc_threshold = cfg.risk.min_ensemble_auc

                        if new_model_auc < min_ensemble_auc_threshold:
                            message = f"[{sym}] <b>CRITICAL:</b> NEW MODEL REJECTED! Its AUC ({new_model_auc:.4f}) is below absolute minimum threshold ({min_ensemble_auc_threshold:.4f}). Trading for this symbol will be blocked."
                            logger.critical(message)
                            notifier.send_message(message, level="CRITICAL")
                            # Keep old model, but block trading for this symbol
                            trading_blocked_by_low_new_model_auc[sym] = True
                        if new_model_auc >= current_active_auc + min_auc_improvement:
                            ens_per_symbol[sym] = new_ens # Replace with the new, better model
                            active_model_auc[sym] = new_model_auc
                            trading_blocked_by_low_new_model_auc[sym] = False # Ensure trading is not blocked
                            message = f"[{sym}] New model (AUC={new_model_auc:.4f}) accepted. Improvement over old (AUC={current_active_auc:.4f})."
                            logger.info(message)
                            notifier.send_message(message, level="INFO")
                        else:
                            message = f"[{sym}] New model (AUC={new_model_auc:.4f}) not significantly better than current (AUC={current_active_auc:.4f}). Keeping current model."
                            logger.warning(message)
                            notifier.send_message(message, level="WARNING")
                            trading_blocked_by_low_new_model_auc[sym] = False # Ensure trading is not blocked
                            # The old model (ens_per_symbol[sym]) remains active
                            # No need to update active_model_auc as it's still the same
                        
                        live_monitor.update_ensemble_auc(active_model_auc[sym]) # Update monitor with AUC of the model that is actually active
                        del retraining_processes[sym]
                        retraining_status[sym] = False # Clear flag
                        logger.info(f"[{sym}] Model handling complete.")

                    # --- Fetch latest bar data using the new pipeline ---
                    feature_cfg = feature_cfg_per_symbol[sym]
                    data, X, y = get_training_data(cfg, sym, feature_cfg=feature_cfg, count=500, source="mt5")
                    if data.empty:
                        continue

                    X_per_symbol[sym] = X
                    latest_bar_time = data.index[-1]
                    if last_bar_time[sym] == latest_bar_time:
                        continue  # skip if no new bar
                    last_bar_time[sym] = latest_bar_time
                    bar_counters[sym] += 1
                    logger.info(f"[{sym}] New bar detected at {latest_bar_time}")

                    # --- Safe Incremental Retraining (now in background) ---
                    if bar_counters[sym] % cfg.retrain_every_bars == 0:
                        if sym not in retraining_processes:
                            logger.info(f"[{sym}] Triggering safe retraining in background.")
                            notifier.send_message(f"[{sym}] Retraining started.", level="INFO") # NEW
                            p = Process(target=run_retraining_in_background, args=(cfg, sym, feature_cfg, dry_run, notifier)) # NEW: Pass notifier
                            p.start()
                            retraining_processes[sym] = p
                            retraining_status[sym] = True # NEW: Set flag
                        else:
                            logger.info(f"[{sym}] Retraining already in progress. Skipping trigger.")

                    # --- Manage existing trades first ---
                    try:
                        atr = float(X["atr_14"].iloc[-1])
                    except Exception:
                        atr = 0.0
                    last_features = X.iloc[[-1]]
                    exe = Execution(ens_per_symbol[sym], risk, dry_run=dry_run, notifier=notifier)
                    
                    # Capture closed trades from manage_trades
                    closed_trades_this_cycle = exe.manage_trades(sym, atr)
                    for trade in closed_trades_this_cycle:
                        live_monitor.add_closed_trade(trade)

                    # --- Update portfolio-level open risk AFTER managing trades ---
                    total_open_risk = sum([pos.get('risk', 0.0) for pos in risk.open_positions_cache.values()])

                    # --- Check trading permission (use cached drawdown + check session) ---
                    if not risk.should_trade(pd.Timestamp.now(), drawdown):
                        logger.info(f"[{sym}] Trade skipped due to drawdown/session rules")
                        continue

                    # --- Execute trade ---
                    if retraining_status[sym]: # Check if retraining is in progress
                        logger.info(f"[{sym}] Trade skipped: Retraining in progress.")
                    elif trading_blocked_by_low_new_model_auc[sym]: # Check if trading is blocked due to low new model AUC
                        logger.info(f"[{sym}] Trade skipped: Trading blocked due to low AUC of newly retrained model.")
                    else:
                        auc_score = getattr(ens_per_symbol[sym], "ensemble_cv_auc_", getattr(ens_per_symbol[sym], "cv_auc_", 0.5))
                        if auc_score < cfg.risk.min_ensemble_auc:
                            message = f"[{sym}] <b>WARNING:</b> Trading blocked due to low ensemble confidence (AUC={auc_score:.4f} < {cfg.risk.min_ensemble_auc:.4f})."
                            logger.warning(message)
                            notifier.send_message(message, level="WARNING")
                        else:
                            result = exe.trade(sym, last_features, atr, auc_score, total_open_risk)
                            if result.ok:
                                logger.info(f"[{sym}] Trade executed: {result.message}")
                            else:
                                logger.info(f"[{sym}] Trade skipped: {result.message}")

                except Exception as e:
                    logger.exception(f"[{sym}] Loop error: {e}")

            # --- Live Dashboard (throttled) ---
            print_dashboard(cfg, risk, ens_per_symbol, X_per_symbol, bar_counter=sum(bar_counters.values()))
            # --- Check for Re-optimization Triggers ---
            triggered, reasons = live_monitor.check_for_reoptimization_trigger(datetime.datetime.now(datetime.timezone.utc))
            if triggered:
                message = f"<b>RE-OPTIMIZATION TRIGGERED!</b> Reasons: {'; '.join(reasons)}"
                logger.critical(message)
                notifier.send_message(message, level="CRITICAL")
                # Here you would add logic to:
                # 1. Set a persistent flag (e.g., write to a file)
                # 2. Send a notification (email, Telegram, etc.)
                # 3. Potentially pause trading in main.py
            
            live_monitor.save_state() # NEW: Save state periodically

            time.sleep(cfg.timeframe_seconds() or 60)

    except KeyboardInterrupt:
        logger.info("=== Stopping MT5 ML Bot ===")
        notifier.send_message("MT5 ML Bot stopped by user (KeyboardInterrupt).", level="WARNING")
        # Optional: clean up any running child processes
        for sym, p in retraining_processes.items():
            if p.is_alive():
                logger.warning(f"[{sym}] Terminating running retraining process due to bot shutdown.")
                p.terminate()
                p.join()
    finally:
        live_monitor.save_state() # NEW: Save state on shutdown
        mt5c.shutdown()
        logger.info("MT5 shutdown complete.")
        notifier.send_message("MT5 ML Bot shutdown complete.", level="INFO")


if __name__ == "__main__":
    # Default to dry-run to be safe; change to False when you are ready.
    run(dry_run=False)
