# src/execution.py
from __future__ import annotations
from dataclasses import dataclass
import MetaTrader5 as mt5  # type: ignore
from loguru import logger
import time
from .ensemble import Ensemble
from .risk import RiskManager
import pandas as pd
import numpy as np
from typing import List, Optional # Import Optional
from backtester import SimPosition # Import SimPosition
from .notifier import TelegramNotifier # NEW import

@dataclass
class OrderResult:
    ok: bool
    ticket: int | None
    message: str

class Execution:
    """ Handles trade decision & order sending with retries + dry-run. """

    def __init__(self, ensemble: Ensemble, risk_manager: RiskManager, dry_run: bool = False, notifier: Optional[TelegramNotifier] = None):
        self.ens = ensemble
        self.risk = risk_manager
        self.dry_run = dry_run
        self.notifier = notifier # NEW
        self.ens = ensemble
        self.risk = risk_manager
        self.dry_run = dry_run

    def _send_order_with_retry(self, request: dict, retries: int = 3, delay: float = 1.0):
        last = None
        for attempt in range(1, retries + 1):
            try:
                result = mt5.order_send(request)
                last = result
                if result is not None and getattr(result, "retcode", None) == getattr(mt5, "TRADE_RETCODE_DONE", 10009):
                    return result
                logger.warning(f"Order send failed attempt {attempt}/{retries}: {result}")
            except Exception as e:
                logger.exception(f"Order send exception attempt {attempt}: {e}")
            time.sleep(delay)
        return last

    def trade(self, symbol: str, X: pd.DataFrame | None = None, atr: float | None = None, auc_score: float | None = 0.5, total_open_risk: float = 0.0) -> OrderResult:
        if X is None or atr is None:
            return OrderResult(False, None, "X or ATR missing")

        # Predict
        try:
            prob_series = self.ens.predict_proba(X.iloc[[-1]])
            # normalize to scalar
            if hasattr(prob_series, "iloc"):
                prob_up = float(prob_series.iloc[0])
            elif isinstance(prob_series, (list, tuple)):
                prob_up = float(prob_series[0])
            else:
                prob_up = float(prob_series)
        except Exception as e:
            logger.exception(f"Prediction failed for {symbol}: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Prediction failed for {symbol}: {e}", level="ERROR")
            return OrderResult(False, None, "Prediction failed")

        # Use optimized threshold if available, otherwise fallback to config
        if self.ens.best_threshold_ is not None:
            threshold = self.ens.best_threshold_
            direction = "long" if prob_up >= threshold else "short"
        else:
            direction = "long" if prob_up >= self.risk.risk_cfg.min_prob_long else "short" if (1 - prob_up) >= self.risk.risk_cfg.min_prob_short else None
        if direction is None:
            return OrderResult(False, None, f"No trade: p_up={prob_up:.3f}")

        account_info = mt5.account_info()
        if not account_info:
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Account info unavailable for {symbol}.", level="ERROR")
            return OrderResult(False, None, "Account info unavailable")
        equity = getattr(account_info, "equity", 0.0)

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Symbol info unavailable for {symbol}.", level="ERROR")
            return OrderResult(False, None, "Symbol info unavailable")

        pip_size = getattr(symbol_info, "point", None)
        contract_size = getattr(symbol_info, "trade_contract_size", 1.0)
        # Basic pip value assumption: pip_value = point * contract_size * lot_size(1) -- warn if unexpected
        pip_value = pip_size * contract_size if pip_size and contract_size else None
        if pip_value is None or pip_value <= 0:
            logger.warning(f"[{symbol}] pip_value computed suspiciously: pip_size={pip_size}, contract_size={contract_size}")
            pip_value = max(1e-6, float(pip_size or 1e-6))

        lots = self.risk.position_size(equity, atr, pip_value, pip_size, auc_score, total_open_risk)
        if lots <= 0:
            return OrderResult(False, None, "Lots <= 0")

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Tick info unavailable for {symbol}.", level="ERROR")
            return OrderResult(False, None, "Tick info unavailable")
        price = float(tick.ask) if direction == "long" else float(tick.bid)

        sl, tp = self.risk.stop_targets(price, atr, direction, auc_score, symbol)
        type_map = {"long": mt5.ORDER_TYPE_BUY, "short": mt5.ORDER_TYPE_SELL}
        deviation_ticks = (float(tick.ask) - float(tick.bid)) if hasattr(tick, "ask") and hasattr(tick, "bid") else 0.0
        deviation = max(10, int(2 * (deviation_ticks) / (pip_size or 1e-6)))

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": type_map[direction],
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": 424242,
            "comment": "ml-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            logger.info(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}")
            if self.notifier: self.notifier.send_message(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}", level="INFO")
            return OrderResult(True, None, "Dry-run prepared")

        res = self._send_order_with_retry(request)
        if res is None or getattr(res, "retcode", None) != getattr(mt5, "TRADE_RETCODE_DONE", 10009):
            error_msg = f"<b>CRITICAL:</b> Order failed for {symbol} after retries: {res}"
            logger.error(error_msg)
            if self.notifier: self.notifier.send_message(error_msg, level="CRITICAL")
            return OrderResult(False, getattr(res, "order", None) if res else None, f"Order failed: {res}")

        ticket = getattr(res, "order", None)
        logger.info(f"Order executed: ticket={ticket}, dir={direction}, lots={lots}, SL={sl}, TP={tp}")
        if self.notifier: self.notifier.send_message(f"<b>TRADE EXECUTED:</b> {direction} {lots} lots of {symbol} at {price:.5f}. SL:{sl:.5f} TP:{tp:.5f}", level="INFO")

        # compute effective risk and store in cache keyed by symbol
        try:
            risk_per_trade = self.risk._get_dynamic_value(self.risk.cfg.dynamic_risk, auc_score, getattr(self.risk.cfg, "risk_per_trade", 0.005))
            risk_amt = equity * risk_per_trade
            sl_distance = max(1e-6, self.risk.cfg.atr_multiplier_sl * atr)
            effective_lots = (risk_amt / (sl_distance * pip_value)) if pip_value and sl_distance else 0.0
            
            # Store comprehensive details for later SimPosition reconstruction
            self.risk.open_positions_cache[symbol] = {
                "risk": float(effective_lots),
                "ticket": ticket,
                "entry_price": price,
                "direction": direction,
                "lots": float(lots),
                "entry_time": datetime.datetime.now(datetime.timezone.utc), # Use current UTC time
                "atr": atr, # ATR at the time of entry
                "entry_auc": auc_score, # AUC at the time of entry
                "risk_fraction": effective_risk, # Effective risk fraction at entry
                "sl": sl, # SL at entry
                "tp": tp, # TP at entry
            }
        except Exception as e:
            logger.warning(f"Could not record open position in cache: {e}")
            if self.notifier: self.notifier.send_message(f"<b>WARNING:</b> Could not record open position in cache for {symbol}: {e}", level="WARNING")

        return OrderResult(True, ticket, "OK")

    def manage_trades(self, symbol: str, atr: float) -> List[SimPosition]:
        try:
            closed_trades = self.risk.manage_open_positions(symbol, atr)
            # NEW: Notify about closed trades
            for trade in closed_trades:
                pnl_msg = "profit" if trade.pnl > 0 else "loss"
                message = f"<b>TRADE CLOSED:</b> {trade.direction} {trade.lots} lots of {trade.symbol}. PnL: {trade.pnl:.2f} ({pnl_msg})."
                logger.info(message)
                if self.notifier: self.notifier.send_message(message, level="INFO")
            return closed_trades
        except Exception as e:
            logger.exception(f"manage_trades error for {symbol}: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> manage_trades error for {symbol}: {e}", level="ERROR")
            return []
