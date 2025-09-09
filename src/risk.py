# src/risk.py
from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from .config import RiskCfg

class RiskManager:
    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg
        self.equity_peak = None

    def position_size(self, equity: float, atr: float, pip_value: float, pip_size: float) -> float:
        """Compute lot size based on equity, ATR stop distance, and risk per trade."""
        risk_amt = equity * self.cfg.risk_per_trade
        sl_distance = self.cfg.atr_multiplier_sl * atr
        if sl_distance <= 0:
            logger.warning(f"SL distance <= 0, cannot compute lots")
            return 0.0
        units = risk_amt / (sl_distance * pip_value)
        lots = max(0.01, min(units * pip_size, 5.0))  # clamp between 0.01 and 5 lots
        logger.debug(f"Position size: equity={equity:.2f}, ATR={atr:.5f}, pip_value={pip_value:.5f}, lots={lots:.2f}")
        return round(lots, 2)

    def stop_targets(self, price: float, atr: float, direction: str):
        """Return initial SL and TP based on ATR multipliers."""
        sl_mult = self.cfg.atr_multiplier_sl
        tp_mult = self.cfg.atr_multiplier_tp
        if direction == "long":
            sl = price - sl_mult * atr
            tp = price + tp_mult * atr
        else:
            sl = price + sl_mult * atr
            tp = price - tp_mult * atr
        logger.debug(f"Stop targets: dir={direction}, price={price:.5f}, SL={sl:.5f}, TP={tp:.5f}")
        return sl, tp

    def should_trade(self, now_local: pd.Timestamp, dd: float) -> bool:
        """Check if trading is allowed (drawdown/session filters)."""
        if dd >= self.cfg.block_on_drawdown:
            logger.info(f"Trading blocked due to drawdown={dd:.3f}")
            return False
        sess = self.cfg.session_filter
        if not sess:
            return True
        start = pd.to_datetime(sess["start"]).time()
        end = pd.to_datetime(sess["end"]).time()
        allowed = start <= now_local.time() <= end
        if not allowed:
            logger.info(f"Trading blocked: current time {now_local.time()} outside session {start}-{end}")
        return allowed

    def manage_open_positions(self, symbol: str, atr: float):
        import MetaTrader5 as mt5
        """
        Adjust stop loss for open positions:
        - Move SL to breakeven at 1R
        - Trail SL by ATR * trailing_atr_mult
        """
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            logger.debug(f"[{symbol}] No open positions")
            return

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"[{symbol}] Symbol info unavailable for managing positions")
            return

        point = symbol_info.point
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"[{symbol}] Tick info unavailable for managing positions")
            return

        for pos in positions:
            entry = pos.price_open
            sl = pos.sl
            direction = "long" if pos.type == mt5.ORDER_TYPE_BUY else "short"

            # Compute profit in pips
            if direction == "long":
                profit_pips = (tick.bid - entry) / point
            else:
                profit_pips = (entry - tick.ask) / point

            # 1R in pips
            one_r = self.cfg.atr_multiplier_sl * atr / point

            new_sl = sl

            # --- move SL to breakeven at 1R ---
            if direction == "long" and profit_pips >= one_r and sl < entry:
                new_sl = entry
            elif direction == "short" and profit_pips >= one_r and sl > entry:
                new_sl = entry

            # --- ATR trailing ---
            trailing_atr = atr * self.cfg.trailing_atr_mult
            if direction == "long":
                new_sl = max(new_sl, tick.bid - trailing_atr)
            else:
                new_sl = min(new_sl, tick.ask + trailing_atr)
            
            # Update position if SL changed
            if new_sl != sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                result = mt5.order_send(request)
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"[{symbol}] SL/TP update failed for ticket {pos.ticket}: {result}")
                else:
                    logger.info(f"[{symbol}] SL/TP updated for ticket {pos.ticket}: new SL={new_sl:.5f}, TP={pos.tp}")
            else:
                logger.debug(f"[{symbol}] No SL adjustment needed for ticket {pos.ticket}")


