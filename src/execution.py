# src/execution.py
from __future__ import annotations
from dataclasses import dataclass
import MetaTrader5 as mt5
from loguru import logger
import numpy as np
from .ensemble import Ensemble
from .risk import RiskManager
import time

@dataclass
class OrderResult:
    ok: bool
    ticket: int | None
    message: str

class Execution:
    """
    Execution handles:
    - Trade decision & order submission
    - Adaptive slippage and retry logic
    - Dry-run support
    """

    def __init__(self, ensemble: Ensemble, risk_manager: RiskManager, dry_run: bool = False):
        self.ens = ensemble
        self.risk = risk_manager
        self.dry_run = dry_run

    # --- Retry wrapper for order submission ---
    def _send_order_with_retry(self, request: dict, retries: int = 3, delay: float = 1.0) -> mt5.OrderSendResult | None:
        for attempt in range(1, retries + 1):
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            logger.warning(f"Order send failed attempt {attempt}/{retries}: {result}")
            time.sleep(delay)
        return result

    # --- Main trade execution ---
    def trade(self, symbol: str, X: np.ndarray | None = None, atr: float | None = None,
              auc_score: float | None = 0.5, total_open_risk: float = 0.0) -> OrderResult:

        if X is None or atr is None:
            return OrderResult(False, None, "X or ATR missing")

        # --- Predict direction ---
        prob_up = self.ens.predict_proba(X.iloc[[-1]]).iloc[0]
        direction = None
        if prob_up >= self.risk.cfg.min_prob_long:
            direction = "long"
        elif (1 - prob_up) >= self.risk.cfg.min_prob_short:
            direction = "short"

        if direction is None:
            return OrderResult(False, None, f"No trade: p_up={prob_up:.3f}")

        # --- Account & symbol info ---
        account_info = mt5.account_info()
        if not account_info:
            return OrderResult(False, None, "Account info unavailable")
        equity = account_info.equity

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return OrderResult(False, None, "Symbol info unavailable")

        pip_size = symbol_info.point
        pip_value = pip_size * symbol_info.trade_contract_size

        # --- Compute lot size ---
        lots = self.risk.position_size(equity, atr, pip_value, pip_size, auc_score, total_open_risk)
        if lots <= 0:
            return OrderResult(False, None, "Lots <= 0")

        # --- Initial SL/TP ---
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if direction == "long" else tick.bid
        sl, tp = self.risk.stop_targets(price, atr, direction, auc_score)

        # --- Prepare order request ---
        type_map = {"long": mt5.ORDER_TYPE_BUY, "short": mt5.ORDER_TYPE_SELL}
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": type_map[direction],
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": max(10, int(2 * (tick.ask - tick.bid) / pip_size)),
            "magic": 424242,
            "comment": "ml-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            logger.info(f"[DRY-RUN] Trade prepared: {direction}, lots={lots}, SL={sl:.5f}, TP={tp:.5f}")
            return OrderResult(True, None, "Dry-run mode")

        # --- Execute with retry ---
        result = self._send_order_with_retry(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, None, f"Order failed after retries: {result}")

        logger.info(f"Order executed: ticket={result.order}, dir={direction}, lots={lots}, SL={sl}, TP={tp}")
        return OrderResult(True, result.order, "OK")

    # --- Manage open positions ---
    def manage_trades(self, symbol: str, atr: float):
        self.risk.manage_open_positions(symbol, atr)
