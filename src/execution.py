# src\execution.py
from __future__ import annotations
from dataclasses import dataclass
import MetaTrader5 as mt5
from loguru import logger
import numpy as np
from .ensemble import Ensemble
from .risk import RiskManager

@dataclass
class OrderResult:
    ok: bool
    ticket: int | None
    message: str

class Execution:
    def __init__(self, ensemble: Ensemble, risk_manager: RiskManager):
        """
        ensemble: trained Ensemble with isotonic calibration
        risk_manager: RiskManager instance handling lot sizing & SL/TP
        """
        self.ens = ensemble
        self.risk = risk_manager

    def trade(self, symbol: str, X: np.ndarray | None = None, atr: float | None = None) -> OrderResult:
        """Make autonomous trade decision and execute."""
        if X is None or atr is None:
            logger.warning(f"[{symbol}] Trade skipped: missing X or ATR")
            return OrderResult(False, None, "X and ATR required")

        prob_up = self.ens.predict_proba(X.iloc[[-1]]).iloc[0]

        # --- decide direction ---
        direction = None
        if prob_up >= self.risk.cfg.min_prob_long:
            direction = "long"
        elif (1 - prob_up) >= self.risk.cfg.min_prob_short:
            direction = "short"

        if direction is None:
            logger.info(f"[{symbol}] No trade executed. p_up={prob_up:.3f}")
            return OrderResult(False, None, f"No trade: p_up={prob_up:.3f}")

        # --- fetch account & symbol info ---
        account_info = mt5.account_info()
        if not account_info:
            logger.error(f"[{symbol}] Account info unavailable")
            return OrderResult(False, None, "Account info unavailable")
        equity = account_info.equity

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] Symbol info unavailable")
            return OrderResult(False, None, "Symbol info unavailable")

        pip_size = symbol_info.point
        pip_value = pip_size * symbol_info.trade_contract_size

        # --- compute lot size ---
        lots = self.risk.position_size(equity, atr, pip_value, pip_size)
        logger.debug(f"[{symbol}] Computed lots={lots:.2f} | equity={equity:.2f}, ATR={atr:.5f}, pip_value={pip_value:.5f}")

        if lots <= 0:
            logger.info(f"[{symbol}] Computed lots <= 0, skipping trade")
            return OrderResult(False, None, "Computed lots <= 0")

        # --- initial SL/TP ---
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if direction == "long" else tick.bid
        sl, tp = self.risk.stop_targets(price, atr, direction)
        logger.debug(f"[{symbol}] Prepared trade: dir={direction}, price={price:.5f}, SL={sl:.5f}, TP={tp:.5f}")

        # --- send order ---
        type_map = {"long": mt5.ORDER_TYPE_BUY, "short": mt5.ORDER_TYPE_SELL}
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": type_map[direction],
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 424242,
            "comment": "ml-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = str(result) if result else "MT5 order_send returned None"
            logger.error(f"[{symbol}] Order failed: {msg}")
            return OrderResult(False, None, msg)

        logger.info(f"[{symbol}] Order executed: ticket={result.order}, dir={direction}, lots={lots}, SL={sl}, TP={tp}")
        return OrderResult(True, result.order, "OK")

    def manage_trades(self, symbol: str, atr: float):
        """Adjust SL for open trades (BE + ATR trailing)."""
        logger.debug(f"[{symbol}] Managing open trades with ATR={atr}")
        self.risk.manage_open_positions(symbol, atr)
