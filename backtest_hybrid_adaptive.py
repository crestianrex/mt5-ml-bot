# backtest_hybrid_adaptive.py — adaptive hybrid backtesting with saved ensembles & incremental retraining
from __future__ import annotations
import pandas as pd
from loguru import logger

from src.config import Cfg
from src.risk import RiskManager
from src.utils import get_training_data, load_ensemble, save_ensemble, setup_logging
import MetaTrader5 as mt5

if not mt5.initialize():
    print("initialize() failed")

# --- Initial Setup ---
setup_logging()

# NOTE: For backtesting, we assume a standard pip size. This is a simplification.
# For a more precise backtest, this could be fetched per-symbol.
PIP_SIZE_ASSUMPTION = 0.0001

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
        self.equity = 10000.0
        self.positions: list[SimPosition] = []
        self.equity_curve = []
        self.bar_counters = {sym: 0 for sym in cfg.symbols}
        self.risks = {sym: RiskManager(cfg.risk) for sym in cfg.symbols}
        
        # Load ensembles for each symbol using the new utility function
        self.ens_per_symbol = {sym: load_ensemble(cfg, sym) for sym in cfg.symbols}
        
        # Get transaction cost from config, default to 0 if not present
        self.cost_in_points = getattr(cfg.risk, 'transaction_cost_pips', 0.0) * PIP_SIZE_ASSUMPTION

    def _update_positions(self, sym, row):
        """Check open positions for SL/TP, calculate PnL including costs, and update equity."""
        for pos in [p for p in self.positions if p.symbol==sym and p.status=="open"]:
            price = row["close"]
            
            # Check if SL or TP is hit
            if (pos.direction=="long" and (price <= pos.sl or price >= pos.tp)) or \
               (pos.direction=="short" and (price >= pos.sl or price <= pos.tp)):
                
                # Calculate gross PnL
                gross_pnl = (price - pos.entry_price) * pos.lots if pos.direction=="long" else (pos.entry_price - price) * pos.lots
                
                # Subtract transaction cost
                transaction_cost = self.cost_in_points * pos.lots
                net_pnl = gross_pnl - transaction_cost
                
                pos.close(price, row.name, net_pnl)
                self.equity += net_pnl

    def run(self):
        for sym in self.cfg.symbols:
            data, X, y = get_training_data(self.cfg, sym)
            if data.empty:
                logger.warning(f"No bars for {sym}, skipping backtest.")
                continue

            ens = self.ens_per_symbol[sym]
            risk_mgr = self.risks[sym]

            for i in range(len(data)):
                # ... (rest of the loop is the same)
                bar_time = data.index[i]
                self.bar_counters[sym] += 1
                last_features = X.iloc[[i]]
                atr = X["atr_14"].iloc[i]

                if self.bar_counters[sym] % self.cfg.retrain_every_bars == 0:
                    window_size = min(self.cfg.history_bars, i + 1)
                    train_data = data.iloc[-window_size:]
                    
                    logger.info(f"[{sym}] Ensemble retraining at {bar_time}...")
                    ens.fit(train_data.drop(columns=["y","close","high","low","volume"]), train_data["y"])
                    
                    save_ensemble(ens, sym)

                prob_up = ens.predict_proba(last_features).iloc[0]
                direction = "long" if prob_up >= risk_mgr.cfg.min_prob_long else "short" if (1 - prob_up) >= risk_mgr.cfg.min_prob_short else None

                self._update_positions(sym, data.iloc[i])

                if direction:
                    lots = risk_mgr.position_size(self.equity, atr, 1.0, PIP_SIZE_ASSUMPTION)
                    if lots > 0:
                        price = data["close"].iloc[i]
                        sl, tp = risk_mgr.stop_targets(price, atr, direction)
                        pos = SimPosition(sym, direction, lots, price, sl, tp, bar_time, atr)
                        self.positions.append(pos)

                eq = self.equity + sum(p.pnl for p in self.positions if p.pnl is not None)
                self.equity_curve.append((bar_time, eq))

        # Close remaining open positions at last price, including transaction costs
        for pos in [p for p in self.positions if p.status=="open"]:
            last_price = data["close"].iloc[-1]
            gross_pnl = (last_price - pos.entry_price) * pos.lots if pos.direction=="long" else (pos.entry_price - last_price) * pos.lots
            transaction_cost = self.cost_in_points * pos.lots
            net_pnl = gross_pnl - transaction_cost
            
            pos.close(last_price, data.index[-1], net_pnl)
            self.equity += net_pnl

        eq_df = pd.DataFrame(self.equity_curve, columns=["time","equity"]).set_index("time")
        trades_df = pd.DataFrame([p.__dict__ for p in self.positions])
        eq_df.to_csv("equity_curve_hybrid_adaptive.csv")
        trades_df.to_csv("trades_hybrid_adaptive.csv")
        logger.info("Hybrid adaptive backtest complete. Results saved to CSV.")
        return trades_df, eq_df

if __name__ == "__main__":
    cfg = Cfg.from_yaml("config.yaml")
    bt = HybridBacktester(cfg)
    trades_df, eq_df = bt.run()
    print("\n--- Trades Summary ---")
    print(trades_df.tail())
    print("\n--- Equity Curve ---")
    print(eq_df.tail())
