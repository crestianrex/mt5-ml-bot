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
from src.risk_controller import RiskController # NEW
import datetime
import json
from typing import Dict, Any
from src.data_manager import DataManager  # NEW: centralized data handler
from src.bandit_warmstart import find_latest_backtest_state, merge_warmstart

import csv # NEW

# --- Initial Setup ---
load_dotenv()
setup_logging()

# --- Metrics Logging Setup ---
METRICS_CSV_FILE = "risk_metrics.csv"
METRICS_HEADERS = [
    "timestamp", "symbol", "event_type", "atr_idx", "min_prob_idx",
    "atr_mult_sl", "atr_mult_tp", "min_prob_long", "min_prob_short",
    "rule_scale", "reward", "equity", "peak_equity", "drawdown", "ensemble_auc"
]

def _initialize_metrics_csv():
    if not os.path.exists(METRICS_CSV_FILE):
        with open(METRICS_CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_HEADERS)
        logger.info(f"Initialized metrics CSV file: {METRICS_CSV_FILE}")

# Call initialization at startup
_initialize_metrics_csv()


def log_metrics_to_csv(data: Dict[str, Any]):
    with open(METRICS_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [data.get(header, "") for header in METRICS_HEADERS]
        writer.writerow(row)

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
            # Correctly find and format positions for the current symbol
            positions_for_symbol = [
                pos for pos in risk.open_positions_cache.values()
                if pos.get('symbol') == sym
            ]
            open_pos_str = ", ".join([
                f"Ticket({p.get('ticket')}, {p.get('direction')}, {p.get('lots')} lots)"
                for p in positions_for_symbol
            ]) if positions_for_symbol else "None"
            logger.info(f"[{sym}] ATR={atr:.5f} | p_up={p_up:.3f} | Open Positions: {open_pos_str}")

def run_retraining_in_background(cfg, sym, feature_cfg, dry_run, notifier):
    """
    A wrapper function to run the entire retraining pipeline in a separate process.
    """
    try:
        # Use DataManager's cached loader (avoids double work and ensures consistent caching)
        data_manager = DataManager(cfg)
        full_data, full_X, full_y = data_manager.load_cached(sym, feature_cfg, count=cfg.retraining_window_bars)

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


def log_startup_summary(cfg: Cfg):
    """ Logs a summary of key configuration settings at startup. """
    logger.info("--- Bot Configuration Summary ---")
    logger.info(f"Symbols: {cfg.symbols}")
    logger.info(f"Timeframe: {cfg.timeframe}")
    logger.info(f"Data Source: {cfg.data_source}")
    logger.info(f"GPU Enabled: {cfg.use_gpu}")

    # Retraining settings
    if cfg.fetch.retrain_time_utc:
        logger.info(f"Retraining Schedule: Daily at {cfg.fetch.retrain_time_utc} UTC")
    else:
        logger.info(f"Retraining Schedule: Every {cfg.retrain_every_bars} bars")
    
    if cfg.retraining_window_bars:
        logger.info(f"Retraining Window: Rolling {cfg.retraining_window_bars} bars")
    else:
        logger.info("Retraining Window: Expanding")

    # Risk and Ensemble settings
    logger.info(f"Max Portfolio Risk: {cfg.risk.max_portfolio_risk}")
    logger.info(f"Drawdown Block Limit: {cfg.risk.block_on_drawdown}")
    
    ensemble_cfg = getattr(cfg, 'ensemble', {})
    logger.info(f"Ensemble Method: {ensemble_cfg.get('method', 'soft_vote') if isinstance(ensemble_cfg, dict) else 'N/A'}")
    logger.info(f"Auto-Threshold Enabled: {ensemble_cfg.get('auto_threshold', False) if isinstance(ensemble_cfg, dict) else 'N/A'}")
    
    # Thompson Sampling settings
    ts_cfg = getattr(cfg, 'thompson_sampling', None)
    if ts_cfg:
        logger.info(f"Thompson Sampling Enabled: {ts_cfg.enabled}")
        if ts_cfg.enabled:
            logger.info(f"  - Contextual Bandit: {ts_cfg.contextual_enabled}")
            logger.info(f"  - Adaptive Grids: {ts_cfg.adaptive_grids_enabled}")
            logger.info(f"  - Bandit Reset: {ts_cfg.bandit_reset_enabled}")
    else:
        logger.info("Thompson Sampling Enabled: N/A")

    logger.info("---------------------------------")


def run(dry_run: bool = False):
    """ Production-ready main loop for hybrid adaptive MT5 ML bot. """
    cfg = Cfg.from_yaml("config.yaml")
    cfg.dashboard_every_bars = getattr(cfg, "dashboard_every_bars", 1)

    if cfg.startup_logging:
        log_startup_summary(cfg)

    logger.info("=== Starting MT5 ML Bot (Hybrid Adaptive) ===")
    logger.info(f"Dry-run mode: {dry_run}")
    logger.info(f"Symbols: {cfg.symbols if hasattr(cfg,'symbols') else []}")

    # Initialize notifier
    notifier = TelegramNotifier(cfg) # NEW: Initialize notifier

    # Initialize DataManager
    data_manager = DataManager(cfg)

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

    # --- Load Ensembles and Feature Configs ---
    ens_per_symbol = {}
    active_model_auc = {} # NEW: To store AUC of currently active model
    feature_cfg_per_symbol = {}
    # Single pass: load ensembles and feature configs once, and bootstrap history via DataManager
    for sym in cfg.symbols:
        ens_per_symbol[sym] = load_ensemble(cfg, sym)
        active_model_auc[sym] = getattr(ens_per_symbol[sym], "ensemble_cv_auc_", getattr(ens_per_symbol[sym], "cv_auc_", 0.5))

        optuna_params = load_optuna_params(sym)
        feature_params = optuna_params.get('features', {}) if optuna_params else {}
        feature_cfg_per_symbol[sym] = FeatureConfig(**feature_params)

        # Bootstrap historical data with caching (chunked fetch if needed)
        logger.info(f"[{sym}] Bootstrapping local history...")
        # support both nested fetch config and legacy cfg.initial_fetch_bars
        fetch_cfg = getattr(cfg, "fetch", None)
        if fetch_cfg is None:
            initial_bars = getattr(cfg, "initial_fetch_bars", getattr(cfg, "history_bars", 30000))
        else:
            # fetch_cfg may be an object or dict depending on your Cfg implementation
            if isinstance(fetch_cfg, dict):
                initial_bars = fetch_cfg.get("initial_fetch_bars", getattr(cfg, "history_bars", 30000))
            else:
                initial_bars = getattr(fetch_cfg, "initial_fetch_bars", getattr(cfg, "history_bars", 30000))

        data_manager.bootstrap_history(sym, initial_bars=initial_bars)

    bar_counters = {sym: 0 for sym in cfg.symbols}
    last_bar_time = {sym: None for sym in cfg.symbols}
    X_per_symbol = {}
    # Warm-start bandit: merge latest backtest priors into live state file BEFORE instantiating RiskController
    try:
        latest_backtest = find_latest_backtest_state(results_dir="results")
        if latest_backtest:
            warm_weight = getattr(getattr(cfg, "thompson_sampling", {}), "warmstart_weight", 1.0)
            logger.info(f"Found backtest bandit state: {latest_backtest}; merging into live state (weight={warm_weight})")
            live_state_path = getattr(getattr(cfg, "thompson_sampling", {}), "state_file", "ts_risk_controller_state.json")
            merge_warmstart(latest_backtest, live_state_path, warmstart_weight=warm_weight)
        else:
            logger.info("No backtest bandit state file found to warm-start.")
    except Exception:
        logger.exception("Warmstart merge failed; continuing without warmstart.")

    # Instantiate risk controller AFTER warmstart merge so it loads the merged state
    risk = RiskManager(cfg, notifier=notifier) # NEW: Pass notifier
    risk_controller = RiskController(cfg, notifier=notifier) # NEW: Instantiate RiskController
    loaded_open_positions = risk_controller.load_state() # Load state again to get open_positions_cache

    # Execution object (single instance)
    exe = Execution(ens_per_symbol, risk, mt5c, dry_run=dry_run, notifier=notifier)
    exe.risk.open_positions_cache.update(loaded_open_positions) # Initialize exe's cache with loaded data

    # NEW: Reconcile open positions with MT5 to ensure accuracy
    exe.reconcile_open_positions_with_mt5()

    retraining_processes = {}
    retraining_status = {sym: False for sym in cfg.symbols} # NEW: Track if retraining is active
    trading_blocked_by_low_new_model_auc = {sym: False for sym in cfg.symbols} # NEW: Track if trading is blocked due to low new model AUC
    last_diagnostics_log_time = 0.0 # NEW: For throttling diagnostics logging
    last_retrain_date = None # NEW: Track last retraining date

    try:
        while True:
            current_loop_time = time.time() # NEW: Capture current time for throttling

            # --- Scheduled Retraining Check ---
            time_to_retrain_today = False
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if cfg.fetch.retrain_time_utc and (last_retrain_date is None or last_retrain_date < now_utc.date()):
                retrain_hour, retrain_minute = map(int, cfg.fetch.retrain_time_utc.split(':'))
                if now_utc.hour > retrain_hour or (now_utc.hour == retrain_hour and now_utc.minute >= retrain_minute):
                    time_to_retrain_today = True
                    last_retrain_date = now_utc.date()

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

            # --- Process closed trades first so bandit gets rewards before opening new trades ---
            closed_trades_this_cycle = exe.check_closed_trades()
            for trade in closed_trades_this_cycle:
                try:
                    live_monitor.add_closed_trade(trade)
                    risk_controller.update_after_trade(trade.symbol, trade)
                    trade_auc = active_model_auc.get(trade.symbol, 0.5)
                    log_metrics_to_csv({
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "symbol": trade.symbol,
                        "event_type": "trade_reward",
                        "atr_idx": trade.atr_idx,
                        "min_prob_idx": trade.min_prob_idx,
                        "reward": trade.profit / max(1.0, equity),
                        "equity": equity,
                        "peak_equity": risk.equity_peak,
                        "drawdown": drawdown,
                        "ensemble_auc": trade_auc
                    })
                except Exception:
                    logger.exception("Error processing closed trade.")

            # --- Per-symbol processing: fetch, update cache, optionally retrain, then make trade decision for that symbol ---
            for sym in cfg.symbols:
                try:
                    # --- Check for finished retraining processes for this symbol ---
                    if sym in retraining_processes and not retraining_processes[sym].is_alive():
                        logger.info(f"[{sym}] Background retraining process finished. Joining process and reloading model.")
                        retraining_processes[sym].join()
                        new_ens = load_ensemble(cfg, sym)
                        new_model_auc = getattr(new_ens, "ensemble_cv_auc_", getattr(new_ens, "cv_auc_", 0.5))
                        current_active_auc = active_model_auc[sym]
                        min_auc_improvement = cfg.risk.min_auc_improvement
                        min_ensemble_auc_threshold = cfg.risk.min_ensemble_auc

                        if new_model_auc < min_ensemble_auc_threshold:
                            message = f"[{sym}] <b>CRITICAL:</b> NEW MODEL REJECTED! Its AUC ({new_model_auc:.4f}) is below absolute minimum threshold ({min_ensemble_auc_threshold:.4f}). Trading for this symbol will be blocked."
                            logger.critical(message)
                            notifier.send_message(message, level="CRITICAL")
                            trading_blocked_by_low_new_model_auc[sym] = True
                        elif new_model_auc >= current_active_auc + min_auc_improvement:
                            ens_per_symbol[sym] = new_ens
                            active_model_auc[sym] = new_model_auc
                            trading_blocked_by_low_new_model_auc[sym] = False
                            message = f"[{sym}] New model (AUC={new_model_auc:.4f}) accepted. Improvement over old (AUC={current_active_auc:.4f})."
                            logger.info(message)
                            notifier.send_message(message, level="INFO")
                        else:
                            message = f"[{sym}] New model (AUC={new_model_auc:.4f}) not significantly better than current (AUC={current_active_auc:.4f}). Keeping current model."
                            logger.warning(message)
                            notifier.send_message(message, level="WARNING")
                            trading_blocked_by_low_new_model_auc[sym] = False

                        live_monitor.update_ensemble_auc(active_model_auc[sym])
                        del retraining_processes[sym]
                        retraining_status[sym] = False
                        logger.info(f"[{sym}] Model handling complete.")

                    # --- Fetch latest bar data using the centralized DataManager pipeline ---
                    feature_cfg = feature_cfg_per_symbol[sym]
                    data, X, y = data_manager.fetch_live(sym, feature_cfg)
                    if data.empty:
                        continue

                    X_per_symbol[sym] = X
                    latest_bar_time = data.index[-1]
                    if last_bar_time[sym] == latest_bar_time:
                        continue  # skip if no new bar
                    last_bar_time[sym] = latest_bar_time
                    bar_counters[sym] += 1
                    logger.info(f"[{sym}] New bar detected at {latest_bar_time}")

                    # Delta append new bars to cache (atomic write)
                    data_manager.append_new_bars(sym, data)

                    # --- Conditional Retraining Logic ---
                    should_retrain = False
                    if time_to_retrain_today:
                        should_retrain = True
                    elif not cfg.fetch.retrain_time_utc:  # Fallback to bar count if time-based is disabled
                        if bar_counters[sym] > 0 and bar_counters[sym] % cfg.retrain_every_bars == 0:
                            should_retrain = True
                    
                    if should_retrain:
                        if sym not in retraining_processes:
                            logger.info(f"[{sym}] Triggering retraining (time-based: {time_to_retrain_today}).")
                            notifier.send_message(f"[{sym}] Retraining started.", level="INFO")
                            p = Process(target=run_retraining_in_background, args=(cfg, sym, feature_cfg, dry_run, notifier))
                            p.start()
                            retraining_processes[sym] = p
                            retraining_status[sym] = True
                        else:
                            logger.info(f"[{sym}] Retraining already in progress. Skipping trigger.")

                    # --- Now handle trading decision for this symbol ---
                    try:
                        atr = float(X["atr_14"].iloc[-1]) if (X is not None and not X.empty) else 0.0
                    except Exception:
                        atr = 0.0
                    last_features = X.iloc[[-1]] if (X is not None and not X.empty) else pd.DataFrame()

                    # Update risk manager peak using current equity
                    risk._update_equity_peak(equity)

                    # Permission checks (drawdown/session)
                    if not risk.should_trade(pd.Timestamp.now(), drawdown):
                        logger.info(f"[{sym}] Trade skipped due to drawdown/session rules")
                        continue

                    # Get dynamic risk params from RiskController
                    auc_score = getattr(ens_per_symbol[sym], "ensemble_cv_auc_", getattr(ens_per_symbol[sym], "cv_auc_", 0.5))
                    context = {
                        "vol": atr,
                        "equity": equity,
                        "peak_equity": risk.equity_peak, 
                        "ensemble_auc": auc_score,
                        "adx": float(last_features["adx"].iloc[0]) if "adx" in last_features.columns else 0.0,
                        "macd_diff": float(last_features["macd_diff"].iloc[0]) if "macd_diff" in last_features.columns else 0.0,
                        "volatility_10": float(last_features["volatility_10"].iloc[0]) if "volatility_10" in last_features.columns else 0.0,
                        "dist_from_ema_200": float(last_features["dist_from_ema_200"].iloc[0]) if "dist_from_ema_200" in last_features.columns else 0.0,
                    }
                    dynamic_risk_params = risk_controller.get_params(sym, context)

                    atr_multiplier_sl = dynamic_risk_params["atr_multiplier_sl"]
                    atr_multiplier_tp = dynamic_risk_params["atr_multiplier_tp"]
                    trailing_atr_mult = dynamic_risk_params["trailing_atr_mult"]
                    min_prob_long = dynamic_risk_params["min_prob_long"]
                    min_prob_short = dynamic_risk_params["min_prob_short"]
                    atr_idx = dynamic_risk_params["atr_idx"]
                    min_prob_idx = dynamic_risk_params["min_prob_idx"]
                    rule_scale = dynamic_risk_params.get("rule_scale", 1.0)

                    # Log chosen parameters
                    log_metrics_to_csv({
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "symbol": sym,
                        "event_type": "param_choice",
                        "atr_idx": atr_idx,
                        "min_prob_idx": min_prob_idx,
                        "atr_mult_sl": atr_multiplier_sl,
                        "atr_mult_tp": atr_multiplier_tp,
                        "min_prob_long": min_prob_long,
                        "min_prob_short": min_prob_short,
                        "rule_scale": rule_scale,
                        "equity": equity,
                        "peak_equity": risk.equity_peak,
                        "drawdown": drawdown,
                        "ensemble_auc": auc_score
                    })

                    # Decision / trade
                    if last_features.empty:
                    # if X_for_ensemble.empty: # Use X_for_ensemble here
                        logger.debug(f"[{sym}] Skipping decision: no features for latest bar")
                        continue
                    prob_up = ens_per_symbol[sym].predict_proba(last_features).iloc[0]
                    
                    # # NEW: Use X_for_ensemble for prediction
                    # prob_up = ens_per_symbol[sym].predict_proba(X_for_ensemble.iloc[[-1]]).iloc[0]

                    if ens_per_symbol[sym].best_threshold_ is not None:
                        threshold = ens_per_symbol[sym].best_threshold_
                        
                        # Corrected logic with a no-trade "dead zone"
                        if prob_up >= threshold:
                            direction = "long"
                        elif prob_up <= (1 - threshold):
                            direction = "short"
                        else:
                            direction = None
                        
                        # Original (buggy) logic - trades on every bar. Uncomment to test.
                        # direction = "long" if prob_up >= threshold else "short"
                    else:
                        # Fallback logic if no auto-threshold is found
                        direction = "long" if prob_up >= min_prob_long else "short" if (1 - prob_up) >= min_prob_short else None

                    if direction:
                        total_open_risk = sum([pos.get('risk', 0.0) for pos in risk.open_positions_cache.values()])
                        risk_per_trade = risk._get_dynamic_value(
                            risk.risk_cfg.dynamic_risk, auc_score, getattr(risk.risk_cfg, "risk_per_trade", 0.005)
                        )
                        max_risk_allowed = max(0.0, float(risk.risk_cfg.max_portfolio_risk) - float(total_open_risk))
                        effective_risk = min(risk_per_trade, max_risk_allowed)
                        exploration_mult = dynamic_risk_params.get("exploration_risk_mult", 1.0)
                        effective_risk *= float(exploration_mult)

                        symbol_info = mt5.symbol_info(sym)
                        if not symbol_info:
                            logger.warning(f"[{sym}] Symbol info unavailable. Skipping position size calculation.")
                            continue

                        pip_size = getattr(symbol_info, "point", None)
                        contract_size = getattr(symbol_info, "trade_contract_size", 1.0)
                        pip_value = pip_size * contract_size if pip_size and contract_size else None
                        if pip_value is None or pip_value <= 0:
                            logger.warning(f"[{sym}] pip_value computed suspiciously: pip_size={pip_size}, contract_size={contract_size}. Skipping position size calculation.")
                            continue

                        lots = risk.position_size(
                            equity, atr, pip_value, pip_size, auc_score, total_open_risk
                        )

                        if lots <= 0:
                            logger.info(f"[{sym}] Trade skipped due to risk limits or position size zero.")
                        else:
                            price = float(mt5.symbol_info_tick(sym).ask) if direction == "long" else float(mt5.symbol_info_tick(sym).bid)
                            sl, tp = risk.stop_targets(price, atr, direction, auc_score, sym, sl_mult=atr_multiplier_sl, tp_mult=atr_multiplier_tp)
                            result = exe.trade(sym, last_features, atr, auc_score, total_open_risk, sl_mult=atr_multiplier_sl, tp_mult=atr_multiplier_tp, atr_idx=atr_idx, min_prob_idx=min_prob_idx)
                            # result = exe.trade(sym, X_for_ensemble, atr, auc_score, total_open_risk, sl_mult=atr_multiplier_sl, tp_mult=atr_multiplier_tp, atr_idx=atr_idx, min_prob_idx=min_prob_idx, X_for_context=X_for_context)
                            if result.ok:
                                logger.info(f"[{sym}] Trade executed: {result.message}")
                            else:
                                logger.info(f"[{sym}] Trade skipped: {result.message}")
                    else:
                        logger.info(f"[{sym}] No trade signal. Probs: (Up: {prob_up:.3f}, Down: {1-prob_up:.3f}) ")
                except Exception:
                    logger.exception(f"Per-symbol loop failed for {sym}")

            # --- Update RiskManager's equity peak for drawdown tracking (done for whole loop) ---
            risk._update_equity_peak(equity)

            # --- Diagnostics Logging ---
            # Throttled logging for RiskController diagnostics
            if current_loop_time - last_diagnostics_log_time >= (cfg.timeframe_seconds() * cfg.dashboard_every_bars):
                ts_diagnostics = risk_controller.diagnostics()
                logger.info(f"RiskController Diagnostics: {json.dumps(ts_diagnostics, indent=2)}")
                last_diagnostics_log_time = current_loop_time

            # --- Live Dashboard (throttled) ---
            print_dashboard(cfg, risk, ens_per_symbol, X_per_symbol, bar_counter=sum(bar_counters.values()))

            time.sleep(cfg.timeframe_seconds() or 60)
            # time.sleep(60)

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
        # save risk_controller state if present
        try:
            open_pos_cache_to_save = {}
            if 'exe' in locals() and exe is not None and hasattr(exe, 'risk') and hasattr(exe.risk, 'open_positions_cache'):
                open_pos_cache_to_save = exe.risk.open_positions_cache
            risk_controller.save_state(open_positions_cache=open_pos_cache_to_save)
        except Exception:
            logger.exception("Failed to save risk_controller state on shutdown.")
        mt5c.shutdown()
        logger.info("MT5 shutdown complete.")
        notifier.send_message("MT5 ML Bot shutdown complete.", level="INFO")


if __name__ == "__main__":
    # Default to dry-run to be safe; change to False when you are ready.
    run(dry_run=False)