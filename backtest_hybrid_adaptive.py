# backtest_hybrid_adaptive.py — adaptive hybrid backtesting with saved ensembles & incremental retraining
from __future__ import annotations
import pandas as pd
from loguru import logger
import os

from src.config import Cfg
from src.risk import RiskManager
from src.utils import get_training_data, load_ensemble, save_ensemble, setup_logging


# --- Initial Setup ---
setup_logging()

# NOTE: For backtesting, we assume a standard pip size. This is a simplification.
# For a more precise backtest, this could be fetched per-symbol.
PIP_SIZE_ASSUMPTION = 0.0001
CONTRACT_SIZE = 100000

class SimPosition:
    """Simulated position for backtesting."""
    def __init__(self, symbol, direction, lots, entry_price, sl, tp, entry_time, atr):
        self.symbol = symbol
        self.direction = direction
        self.lots = lots
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time
        self.exit_time = None
        self.exit_price = None
        self.pnl = None
        self.atr = atr
        self.status = "open"

    def close(self, price, time, pnl):
        self.exit_price = price
        self.exit_time = time
        self.pnl = pnl
        self.status = "closed"

class HybridBacktester:
    """Adaptive hybrid backtester mirroring main_hybrid_adaptive.py logic."""
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.equity = 1000.0
        self.positions: list[SimPosition] = []
        self.equity_curve = []
        self.bar_counters = {sym: 0 for sym in cfg.symbols}
        self.risks = {sym: RiskManager(cfg.risk) for sym in cfg.symbols}
        
        logger.info(f"Initializing backtester with starting equity: {self.equity}")
        self.ens_per_symbol = {sym: load_ensemble(cfg, sym) for sym in cfg.symbols}
        
        self.cost_in_points = getattr(cfg.risk, 'transaction_cost_pips', 0.0) * PIP_SIZE_ASSUMPTION
        if self.cost_in_points > 0:
            logger.info(f"Applying transaction cost: {getattr(cfg.risk, 'transaction_cost_pips', 0.0)} pips per trade.")

    def _manage_trailing_stops(self, sym: str, row: pd.Series, atr: float):
        """Simulated version of the live trailing stop logic."""
        risk_cfg = self.risks[sym].cfg
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
                
                pos.close(price, row.name, net_pnl)
                self.equity += net_pnl
                logger.info(
                    f"[{sym}] Closed {pos.direction} position at {price:.5f} due to {exit_reason}. "
                    f"Entry: {pos.entry_price:.5f}, PnL: {net_pnl:.2f}, Equity: {self.equity:.2f}"
                )

    def run(self):
        logger.info("=== Starting Hybrid Adaptive Backtest ===")
        for sym in self.cfg.symbols:
            logger.info(f"--- Backtesting Symbol: {sym} ---")
            data, X, y = get_training_data(self.cfg, sym, source="csv")
            if data.empty:
                logger.warning(f"No data for {sym}, skipping.")
                continue

            ens = self.ens_per_symbol[sym]
            risk_mgr = self.risks[sym]

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

                # Retrain if needed
                if self.bar_counters[sym] > 0 and self.bar_counters[sym] % self.cfg.retrain_every_bars == 0:
                    window_size = min(self.cfg.history_bars, i + 1)
                    train_data = data.iloc[i - window_size + 1 : i + 1]
                    logger.info(
                        f"[{sym}] Ensemble retraining at {bar_time} using last {len(train_data)} bars..."
                    )
                    ens.fit(train_data.drop(columns=["y","close","high","low","volume"]), train_data["y"])
                    save_ensemble(ens, sym)

                # Check ensemble confidence before trading
                if ens.ensemble_cv_auc_ is not None and ens.ensemble_cv_auc_ < risk_mgr.cfg.min_ensemble_auc:
                    logger.info(f"[{sym}] Trading blocked due to low ensemble confidence (AUC={ens.ensemble_cv_auc_:.4f} < {risk_mgr.cfg.min_ensemble_auc:.4f}).")
                    self.equity_curve.append((bar_time, self.equity)) # Append current equity even if no trade
                    continue # Skip trade decision for this bar

                # Decide on new trades
                prob_up = ens.predict_proba(last_features).iloc[0]
                direction = "long" if prob_up >= risk_mgr.cfg.min_prob_long else "short" if (1 - prob_up) >= risk_mgr.cfg.min_prob_short else None

                if direction:
                    if any(p.symbol == sym and p.status == "open" for p in self.positions):
                        logger.debug(f"[{sym}] Skipping new trade; existing position is open.")
                    else:
                        lots = risk_mgr.position_size(self.equity, atr, 10.0, PIP_SIZE_ASSUMPTION)
                        if lots > 0:
                            price = current_row["close"]
                            sl, tp = risk_mgr.stop_targets(price, atr, direction)
                            # Store the ATR at time of trade for breakeven calculations
                            pos = SimPosition(sym, direction, lots, price, sl, tp, bar_time, atr)
                            self.positions.append(pos)
                            logger.info(
                                f"[{sym}] Opened {direction} position at {price:.5f}. "
                                f"Lots: {lots:.2f}, SL: {sl:.5f}, TP: {tp:.5f}"
                            )
                else:
                    logger.debug(f"[{sym}] No trade signal. Probs: (Up: {prob_up:.3f}, Down: {1-prob_up:.3f})")

                self.equity_curve.append((bar_time, self.equity))
            
            logger.info(f"--- Completed Backtest for Symbol: {sym} ---")

            # --- Close any positions left open for the current symbol ---
            logger.info(f"Closing any remaining open positions for {sym}...")
            for pos in [p for p in self.positions if p.symbol == sym and p.status == "open"]:
                last_row = data.iloc[-1]
                last_price = last_row["close"]
                gross_pnl = ((last_price - pos.entry_price) * pos.lots * CONTRACT_SIZE) if pos.direction=="long" else ((pos.entry_price - last_price) * pos.lots * CONTRACT_SIZE)
                transaction_cost = self.cost_in_points * pos.lots * CONTRACT_SIZE
                net_pnl = gross_pnl - transaction_cost
                
                pos.close(last_price, last_row.name, net_pnl)
                self.equity += net_pnl
                logger.info(
                    f"[{pos.symbol}] Force-closed open {pos.direction} position at final price {last_price:.5f}. "
                    f"PnL: {net_pnl:.2f}, Final Equity: {self.equity:.2f}"
                )

        # --- Final Reporting (outside the symbol loop) ---
        eq_df = pd.DataFrame(self.equity_curve, columns=["time","equity"]).set_index("time")
        trades_df = pd.DataFrame([p.__dict__ for p in self.positions])
        
        # Create a results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        eq_df.to_csv("results/equity_curve_hybrid_adaptive.csv")
        trades_df.to_csv("results/trades_hybrid_adaptive.csv")
        
        logger.info(f"=== Hybrid Adaptive Backtest Complete. Final Equity: {self.equity:.2f} ===")
        logger.info("Results saved to 'results/' directory.")
        return trades_df, eq_df

if __name__ == "__main__":
    cfg = Cfg.from_yaml("config.yaml")
    bt = HybridBacktester(cfg)
    trades_df, eq_df = bt.run()
    print("\n--- Trades Summary ---")
    print(trades_df.tail())
    print("\n--- Equity Curve ---")
    print(eq_df.tail())
