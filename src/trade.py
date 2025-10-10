# src/trade.py
from __future__ import annotations

class SimPosition:
    """Simulated position for backtesting."""
    def __init__(self, symbol, direction, lots, entry_price, sl, tp, entry_time, atr, entry_auc, risk_fraction, atr_idx: int = -1, min_prob_idx: int = -1, adx: float = 0.0, macd_diff: float = 0.0, volatility_10: float = 0.0, dist_from_ema_200: float = 0.0):
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
        self.entry_auc = entry_auc
        self.risk_fraction = risk_fraction
        self.atr_idx = atr_idx # NEW
        self.min_prob_idx = min_prob_idx # NEW
        self.exit_equity = None # NEW
        self.adx = adx
        self.macd_diff = macd_diff
        self.volatility_10 = volatility_10
        self.dist_from_ema_200 = dist_from_ema_200

    def close(self, price, time, pnl, exit_equity: float):
        self.exit_price = price
        self.exit_time = time
        self.pnl = pnl
        self.status = "closed"
        self.exit_equity = exit_equity # NEW
