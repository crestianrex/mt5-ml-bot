# backtester.py
from __future__ import annotations
import pandas as pd
from loguru import logger
import os
import datetime
import copy
import quantstats as qs
import optuna
import numpy as np # Added numpy import

from src.config import Cfg
from src.features import FeatureConfig
from src.risk import RiskManager
from src.utils import get_training_data, load_ensemble, save_ensemble, setup_logging, safe_retrain_ensemble, load_optuna_params
from src.trade import SimPosition
from src.risk_controller import RiskController # NEW

# NOTE: For backtesting, we assume a standard pip size. This is a simplification.
# For a more precise backtest, this could be fetched per-symbol.
PIP_SIZE_ASSUMPTION = 0.0001
CONTRACT_SIZE = 100000

class HybridBacktester:
    """Adaptive hybrid backtester mirroring main_hybrid_adaptive.py logic."""
    def _count_consecutive_losses_backtest(self) -> int:
        closed_trades = sorted([p for p in self.positions if p.status == "closed" and p.pnl is not None], key=lambda p: p.exit_time)
        if not closed_trades:
            return 0
        
        count = 0
        for trade in reversed(closed_trades):
            if trade.pnl < 0: # Assuming negative PnL means loss
                count += 1
            else:
                break
        return count

    def __init__(self, cfg: Cfg):
        self.logged_low_confidence = set()
        self.logged_skips = set()
        self.cfg = cfg
        self.equity = cfg.initial_equity
        self.initial_equity = cfg.initial_equity # Store initial equity for drawdown pruning
        self.positions: list[SimPosition] = []
        self.equity_curve = []
        self.bar_counters = {sym: 0 for sym in cfg.symbols}
        self.risk_manager = RiskManager(cfg)
        self.risk_controller = RiskController(cfg) # NEW: Instantiate RiskController
        self.ts_param_history = [] # NEW: To store Thompson Sampling parameter evolution
        self.save_state_every_bars = getattr(cfg, "save_ts_state_every_bars", 500)
        ts_ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.backtest_ts_state_file = f"results/ts_risk_controller_state_backtest_{ts_ts}.json"
        self.ts_history_csv = f"results/ts_param_evolution_backtest_{ts_ts}.csv"
        os.makedirs("results", exist_ok=True)
        
        logger.info(f"Initializing backtester with starting equity: {self.equity}")
        self.ens_per_symbol = {sym: load_ensemble(cfg, sym) for sym in cfg.symbols}
        
        self.cost_in_points = getattr(cfg.risk, 'transaction_cost_pips', 0.0) * PIP_SIZE_ASSUMPTION
        if self.cost_in_points > 0:
            logger.info(f"Applying transaction cost: {getattr(cfg.risk, 'transaction_cost_pips', 0.0)} pips per trade.")

    def _manage_trailing_stops(self, sym: str, row: pd.Series, atr: float):
        """Simulated version of the live trailing stop logic."""
        risk_cfg = self.risk_manager.risk_cfg
        if not (risk_cfg.breakeven_at_1R or risk_cfg.trailing_atr_mult > 0):
            return # No trailing logic enabled

        for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
            price = row["close"]
            new_sl = pos.sl

            # --- Breakeven Logic ---
            if risk_cfg.breakeven_at_1R:
                one_r_price_move = risk_cfg.atr_multiplier_sl * pos.atr
                if pos.direction == "long" and price >= pos.entry_price + one_r_price_move and pos.sl < pos.entry_price:
                    new_sl = pos.entry_price
                    logger.info(f"[{sym}] Moving SL to breakeven for long position at {new_sl:.5f}")
                elif pos.direction == "short" and price <= pos.entry_price - one_r_price_move and pos.sl > pos.entry_price:
                    new_sl = pos.entry_price
                    logger.info(f"[{sym}] Moving SL to breakeven for short position at {new_sl:.5f}")

            # --- ATR Trailing Logic ---
            if risk_cfg.trailing_atr_mult > 0:
                trailing_atr_dist = atr * risk_cfg.trailing_atr_mult
                if pos.direction == "long":
                    potential_new_sl = price - trailing_atr_dist
                    if potential_new_sl > new_sl:
                        new_sl = potential_new_sl
                        logger.debug(f"[{sym}] Trailing SL for long position to {new_sl:.5f}")
                else: # Short position
                    potential_new_sl = price + trailing_atr_dist
                    if potential_new_sl < new_sl:
                        new_sl = potential_new_sl
                        logger.debug(f"[{sym}] Trailing SL for short position to {new_sl:.5f}")
            
            pos.sl = new_sl

    def _update_positions(self, sym, row):
        """Check open positions for SL/TP, calculate PnL, and update equity."""
        closed_trades_this_cycle = [] # NEW: Collect trades closed in this cycle
        for pos in [p for p in self.positions if p.symbol==sym and p.status=="open"]:
            price = row["close"]
            exit_reason = None
            
            if pos.direction == "long":
                if price <= pos.sl:
                    exit_reason = "Stop Loss"
                elif price >= pos.tp:
                    exit_reason = "Take Profit"
            elif pos.direction == "short":
                if price >= pos.sl:
                    exit_reason = "Stop Loss"
                elif price <= pos.tp:
                    exit_reason = "Take Profit"

            if exit_reason:
                gross_pnl = ((price - pos.entry_price) * pos.lots * CONTRACT_SIZE) if pos.direction == "long" else ((pos.entry_price - price) * pos.lots * CONTRACT_SIZE)
                transaction_cost = self.cost_in_points * pos.lots * CONTRACT_SIZE
                net_pnl = gross_pnl - transaction_cost
                
                # Pass current equity to pos.close for reward normalization
                pos.close(price, row.name, net_pnl, self.equity + net_pnl) # NEW: Pass exit_equity
                self.equity += net_pnl
                closed_trades_this_cycle.append(pos) # NEW: Add to list
                logger.info(
                    f"[{sym}] Closed {pos.direction} position at {price:.5f} due to {exit_reason}. "
                    f"Entry: {pos.entry_price:.5f}, PnL: {net_pnl:.2f}, Equity: {self.equity:.2f}"
                )
        
        # NEW: Update RiskController for each closed trade
        for trade in closed_trades_this_cycle:
            self.risk_controller.update_after_trade(sym, trade)
    
    def _perform_retraining(self, sym: str, bar_time: pd.Timestamp, i: int, data: pd.DataFrame, X: pd.DataFrame):
        """
        Handles the logic for retraining the model.
        """
        if self.bar_counters[sym] > 0 and self.bar_counters[sym] % self.cfg.retrain_every_bars == 0:
            window_size = min(self.cfg.history_bars, i + 1)

            # --- FIX: Guard against retraining with insufficient data ---
            # A safe threshold to ensure enough samples for cross-validation.
            # This value should be comfortably larger than the sum of CV splits and min samples per model.
            MIN_BARS_FOR_RETRAIN = 0
            if window_size < MIN_BARS_FOR_RETRAIN:
                logger.warning(f"[{sym}] Skipping retraining at {bar_time}: not enough data in window ({window_size} < {MIN_BARS_FOR_RETRAIN} bars).")
                return self.ens_per_symbol[sym]

            train_data = data.iloc[i - window_size + 1: i + 1]
            logger.info(
                f"[{sym}] Ensemble retraining at {bar_time} using last {len(train_data)} bars..."
            )

            ens_old = self.ens_per_symbol[sym]
            
            # Use the shared safe_retrain_ensemble function
            # IMPORTANT: A dry_run=True flag should be added here to prevent overwriting prod models.
            ens_new = safe_retrain_ensemble(self.cfg, sym, ens_old, train_data[X.columns], train_data["y"], train_data["close"] if "close" in train_data.columns else None, dry_run=True)
            
            # Update the ensemble in the backtester's state
            self.ens_per_symbol[sym] = ens_new
            
        return self.ens_per_symbol[sym]

    def _process_bar(self, sym: str, data: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame, trial: optuna.Trial | None = None, pruning_interval: int = 0):
        """Processes each bar of data for a given symbol."""
        ens = self.ens_per_symbol[sym]
        risk_mgr = self.risk_manager

        logger.info(f"Processing {len(data)} bars for {sym}...")
        for i in range(20, len(data)):
            bar_time = data.index[i]
            current_row = data.iloc[i]
            self.bar_counters[sym] += 1
            last_features = X.iloc[[i]]
            atr = X["atr_14"].iloc[i]

            # Manage existing positions first
            self._manage_trailing_stops(sym, current_row, atr)
            self._update_positions(sym, current_row)

            # --- Drawdown and Cooldown Check ---
            risk_mgr._update_equity_peak(self.equity)
            if risk_mgr._drawdown_exceeded(self.equity):
                if risk_mgr.cooldown_until is None:  # Only trigger if not already in cooldown
                    logger.warning(f"[{sym}][{bar_time}] Drawdown threshold exceeded. Triggering cooldown.")
                    now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
                    risk_mgr._trigger_cooldown(now=now_utc)

            # --- Consecutive Loss Check ---
            max_losses = getattr(risk_mgr.watchdog_cfg, "max_consecutive_losses", None)
            if max_losses is not None and max_losses > 0:
                consecutive_losses = self._count_consecutive_losses_backtest()
                if consecutive_losses >= max_losses:
                    if risk_mgr.cooldown_until is None:
                        logger.warning(f"[{sym}][{bar_time}] Watchdog: consecutive losses {consecutive_losses} >= threshold {max_losses}. Triggering cooldown.")
                        now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
                        risk_mgr._trigger_cooldown(now=now_utc)

            now_utc = bar_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
            if risk_mgr.cooldown_active(now=now_utc):
                logger.info(f"[{sym}][{bar_time}] Trading blocked: watchdog cooldown active.")
                self.equity_curve.append((bar_time, self.equity))
                # --- Pruning Check (if in tuning mode) ---
                if trial and pruning_interval > 0 and (i % pruning_interval == 0) and self.cfg.symbols.index(sym) == 0:
                    current_returns = pd.Series([eq for _, eq in self.equity_curve]).pct_change().dropna()
                    if not current_returns.empty:
                        intermediate_sharpe = 0.0
                        if current_returns.std() != 0:
                            timeframe_minutes = self.cfg.timeframe_minutes()
                            if timeframe_minutes is not None:
                                annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
                                intermediate_sharpe = current_returns.mean() / current_returns.std() * annualization_factor
                        trial.report(intermediate_sharpe, i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                continue

            # Retrain if needed
            ens = self._perform_retraining(sym, bar_time, i, data, X)

            # Check ensemble confidence before trading
            if ens.ensemble_cv_auc_ is not None and ens.ensemble_cv_auc_ < risk_mgr.risk_cfg.min_ensemble_auc:
                if sym not in self.logged_low_confidence:
                    logger.info(f"[{sym}] Trading blocked due to low ensemble confidence (AUC={ens.ensemble_cv_auc_:.4f} < {risk_mgr.risk_cfg.min_ensemble_auc:.4f}).")
                    self.logged_low_confidence.add(sym)
                self.equity_curve.append((bar_time, self.equity))
                # --- Pruning Check (if in tuning mode) ---
                if trial and pruning_interval > 0 and (i % pruning_interval == 0) and self.cfg.symbols.index(sym) == 0:
                    current_returns = pd.Series([eq for _, eq in self.equity_curve]).pct_change().dropna()
                    if not current_returns.empty:
                        intermediate_sharpe = 0.0
                        if current_returns.std() != 0:
                            timeframe_minutes = self.cfg.timeframe_minutes()
                            if timeframe_minutes is not None:
                                annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
                                intermediate_sharpe = current_returns.mean() / current_returns.std() * annualization_factor
                        trial.report(intermediate_sharpe, i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                continue
            else:
                if sym in self.logged_low_confidence:
                    self.logged_low_confidence.remove(sym)

            # Decide on new trades
            prob_up = ens.predict_proba(last_features).iloc[0]

            # NEW: Get dynamic risk parameters from RiskController
            context = {
                "vol": atr,
                "equity": self.equity,
                "peak_equity": self.risk_manager.peak_equity,
                "ensemble_auc": ens.ensemble_cv_auc_, # Pass current model confidence
                "adx": float(last_features["adx"].iloc[0]) if "adx" in last_features.columns else 0.0,
                "macd_diff": float(last_features["macd_diff"].iloc[0]) if "macd_diff" in last_features.columns else 0.0,
                "volatility_10": float(last_features["volatility_10"].iloc[0]) if "volatility_10" in last_features.columns else 0.0,
                "dist_from_ema_200": float(last_features["dist_from_ema_200"].iloc[0]) if "dist_from_ema_200" in last_features.columns else 0.0,
            }
            dynamic_risk_params = self.risk_controller.get_params(sym, context)

            atr_multiplier_sl = dynamic_risk_params["atr_multiplier_sl"]
            atr_multiplier_tp = dynamic_risk_params["atr_multiplier_tp"]
            trailing_atr_mult = dynamic_risk_params["trailing_atr_mult"]
            min_prob_long = dynamic_risk_params["min_prob_long"]
            min_prob_short = dynamic_risk_params["min_prob_short"]
            atr_idx = dynamic_risk_params["atr_idx"]
            min_prob_idx = dynamic_risk_params["min_prob_idx"]

            if ens.best_threshold_ is not None:
                threshold = ens.best_threshold_
                direction = "long" if prob_up >= threshold else "short"
            else:
                direction = "long" if prob_up >= min_prob_long else "short" if (1 - prob_up) >= min_prob_short else None

            if direction:
                total_open_risk = sum(p.risk_fraction for p in self.positions if p.status == "open")
                risk_per_trade = risk_mgr._get_dynamic_value(
                    risk_mgr.risk_cfg.dynamic_risk, ens.ensemble_cv_auc_, getattr(risk_mgr.risk_cfg, "risk_per_trade", 0.005)
                )
                max_risk_allowed = max(0.0, float(risk_mgr.risk_cfg.max_portfolio_risk) - float(total_open_risk))
                effective_risk = min(risk_per_trade, max_risk_allowed)
                # apply exploration multiplier if present
                exploration_mult = dynamic_risk_params.get("exploration_risk_mult", 1.0)
                effective_risk *= float(exploration_mult)

                lots = risk_mgr.position_size(
                    self.equity, atr, 10.0, PIP_SIZE_ASSUMPTION, ens.ensemble_cv_auc_, total_open_risk
                )

                if lots > 0:
                    price = current_row["close"]
                    # Use dynamic SL/TP multipliers
                    sl, tp = risk_mgr.stop_targets(price, atr, direction, ens.ensemble_cv_auc_, sym, sl_mult=atr_multiplier_sl, tp_mult=atr_multiplier_tp)
                    pos = SimPosition(
                        sym, direction, lots, price, sl, tp, bar_time, atr, ens.ensemble_cv_auc_, effective_risk,
                        atr_idx=atr_idx, min_prob_idx=min_prob_idx # NEW: Store discrete choices
                    )
                    self.positions.append(pos)
                    logger.info(
                        f"[{sym}][{bar_time}] Opened {direction} position at {price:.5f}. "
                        f"Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}, AUC: {ens.ensemble_cv_auc_:.4f}, Risk: {effective_risk:.4f}"
                    )
                else:
                    logger.info(f"[{sym}] Trade skipped due to risk limits or position size zero.")
            else:
                logger.info(f"[{sym}] No trade signal. Probs: (Up: {prob_up:.3f}, Down: {1-prob_up:.3f}) ")

            self.equity_curve.append((bar_time, self.equity))



            # Log chosen parameters and diagnostics for Thompson Sampling analysis
            ts_diagnostics = self.risk_controller.diagnostics()
            self.ts_param_history.append({
                "bar_time": bar_time,
                "symbol": sym,
                "atr_idx": atr_idx,
                "min_prob_idx": min_prob_idx,
                "diagnostics": ts_diagnostics # Store full diagnostics for later parsing
            })
            
            # Periodic persistence of Thompson state + CSV (avoid overwriting prod state)
            total_bars = sum(self.bar_counters.values())
            if total_bars % self.save_state_every_bars == 0:
                # write to the backtest-specific file so you don't overwrite live/prod state
                self._persist_bandit_state(force_path=self.backtest_ts_state_file)


            # --- Pruning Check (if in tuning mode) ---
            if trial and pruning_interval > 0 and (i % pruning_interval == 0) and self.cfg.symbols.index(sym) == 0:
                current_returns = pd.Series([eq for _, eq in self.equity_curve]).pct_change().dropna()
                if not current_returns.empty:
                    intermediate_sharpe = 0.0
                    if current_returns.std() != 0:
                        timeframe_minutes = self.cfg.timeframe_minutes()
                        if timeframe_minutes is not None:
                            annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
                            intermediate_sharpe = current_returns.mean() / current_returns.std() * annualization_factor
                    trial.report(intermediate_sharpe, i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

    def _generate_results(self):
        """Generates and saves the backtesting results."""
        eq_df = pd.DataFrame(self.equity_curve, columns=["time", "equity"]).set_index("time")
        trades_df = pd.DataFrame([p.__dict__ for p in self.positions])

        os.makedirs("results", exist_ok=True)
        symbol_str = "_".join([s.replace('#', '') for s in self.cfg.symbols])

        eq_df.to_csv(f"results/equity_curve_{symbol_str}_hybrid_adaptive.csv")
        trades_df.to_csv(f"results/trades_{symbol_str}_hybrid_adaptive.csv")

        # Generate quantstats report
        try:
            returns = eq_df["equity"].pct_change().dropna()
            returns.index = pd.to_datetime(returns.index)
            report_path = f"results/report_{symbol_str}_hybrid_adaptive.html"
            qs.reports.html(returns, output=report_path, title=f"{symbol_str} Hybrid Adaptive Strategy")
            logger.info(f"QuantStats report saved to {report_path}")
        except Exception as e:
            logger.exception(f"Failed to generate QuantStats report: {e}")

        logger.info(f"=== Hybrid Adaptive Backtest Complete. Final Equity: {self.equity:.2f} ===")
        logger.info("Results saved to 'results/' directory.")

        # NEW: Generate Thompson Sampling parameter evolution report
        if self.ts_param_history:
            ts_df = pd.DataFrame(self.ts_param_history)
            ts_report_path = f"results/ts_param_evolution_{symbol_str}_hybrid_adaptive.csv"
            ts_df.to_csv(ts_report_path, index=False)
            logger.info(f"Thompson Sampling parameter evolution report saved to {ts_report_path}")

        return trades_df, eq_df

    def _persist_bandit_state(self, force_path: str | None = None):
        """
        Persist RiskController state and ts_param_history CSV.
        If force_path supplied, temporarily write the TS JSON to that path.
        """
        try:
            # optionally override cfg path just for this save
            original_state_file = None
            if force_path:
                original_state_file = self.cfg.thompson_sampling.state_file
                self.cfg.thompson_sampling.state_file = force_path

            # Save RiskController internal JSON (uses RiskController.save_state)
            try:
                self.risk_controller.save_state()
            except Exception:
                logger.exception("Failed to save RiskController state with risk_controller.save_state()")

            # Save human-readable CSV of ts history for offline analysis
            if self.ts_param_history:
                import pandas as pd
                df = pd.DataFrame(self.ts_param_history)
                df.to_csv(self.ts_history_csv, index=False)

            logger.info(f"Persisted bandit state to {self.cfg.thompson_sampling.state_file} and CSV to {self.ts_history_csv}")
        except Exception as e:
            logger.exception(f"_persist_bandit_state failed: {e}")
        finally:
            if force_path and original_state_file is not None:
                # restore original
                self.cfg.thompson_sampling.state_file = original_state_file


    def run(self, trial: optuna.Trial | None = None, pruning_interval: int = 0):
        logger.info("=== Starting Hybrid Adaptive Backtest ===")

        for sym in self.cfg.symbols:
            logger.info(f"--- Backtesting Symbol: {sym} ---")
            
            # Load best feature params from optuna study
            optuna_params = load_optuna_params(sym)
            feature_params = optuna_params.get('features', {}) if optuna_params else {}
            feature_cfg = FeatureConfig(**feature_params)

            data, X, y = get_training_data(self.cfg, sym, feature_cfg=feature_cfg, source=self.cfg.data_source)
            if data.empty:
                logger.warning(f"No data for {sym}, skipping.")
                continue

            self._process_bar(sym, data, X, y, trial, pruning_interval)

            logger.info(f"--- Completed Backtest for Symbol: {sym} ---")

            # --- Close any positions left open for the current symbol ---
            logger.info(f"Closing any remaining open positions for {sym}...")
            for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
                last_row = data.iloc[-1]
                last_price = last_row["close"]
                gross_pnl = ((last_price - pos.entry_price) * pos.lots * CONTRACT_SIZE) if pos.direction == "long" else ((pos.entry_price - last_price) * pos.lots * CONTRACT_SIZE)
                transaction_cost = self.cost_in_points * pos.lots * CONTRACT_SIZE
                net_pnl = gross_pnl - transaction_cost

                pos.close(last_price, last_row.name, net_pnl, self.equity + net_pnl)
                self.equity += net_pnl
                logger.info(
                    f"[{pos.symbol}] Force-closed open {pos.direction} position at final price {last_price:.5f}. "
                    f"PnL: {net_pnl:.2f}, Final Equity: {self.equity:.2f}"
                )

        return self._generate_results()

if __name__ == "__main__":
    cfg = Cfg.from_yaml("config.yaml")
    setup_logging(level=cfg.logging["level"], to_file=cfg.logging["to_file"], rotate=cfg.logging["rotate"], retention=cfg.logging["retention"])
    
    cfg.thompson_sampling.state_file = f"ts_risk_controller_state_backtest_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"

    
    bt = HybridBacktester(cfg)
    try:
        trades_df, eq_df = bt.run()
        print("\n--- Trades Summary ---")
        print(trades_df.tail())
        print("\n--- Equity Curve ---")
        print(eq_df.tail())
    finally:
        try:
            bt._persist_bandit_state(force_path=bt.backtest_ts_state_file)
        except Exception:
            logger.exception("Final persist of bandit state failed.")