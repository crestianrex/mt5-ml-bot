# src/risk.py
from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from .config import RiskCfg

class RiskManager:
    """
    RiskManager handles:
    - Dynamic position sizing based on model confidence (AUC)
    - SL/TP calculations with dynamic ATR multipliers
    - Portfolio-level exposure cap
    - Breakeven and ATR trailing for open positions
    """

    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg
        self.equity_peak = None
        self.open_positions_cache = {}  # symbol -> {"risk": float, "ticket": int}

    # --- Dynamic value calculation ---
    def _get_dynamic_value(self, dynamic_cfg: dict, auc_score: float, default_val: float) -> float:
        if not dynamic_cfg or not dynamic_cfg.get("enabled"):
            return default_val

        auc_floor = dynamic_cfg.get("auc_floor", 0.55)
        auc_ceiling = dynamic_cfg.get("auc_ceiling", 0.65)
        base_val = dynamic_cfg.get("base_risk") or dynamic_cfg.get("base_tp_mult")
        max_val = dynamic_cfg.get("max_risk") or dynamic_cfg.get("max_tp_mult")

        clamped_auc = np.clip(auc_score, auc_floor, auc_ceiling)
        val = base_val + (clamped_auc - auc_floor) * (max_val - base_val) / max(auc_ceiling - auc_floor, 1e-6)
        logger.debug(f"Dynamic value calc: AUC={auc_score:.4f}, Clamped={clamped_auc:.4f}, Value={val:.4f}")
        return val

    # --- Position sizing with portfolio exposure cap ---
    def position_size(self, equity: float, atr: float, pip_value: float, pip_size: float,
                      auc_score: float, total_open_risk: float = 0.0) -> float:
        risk_per_trade = self._get_dynamic_value(self.cfg.dynamic_risk, auc_score,
                                                 getattr(self.cfg, 'risk_per_trade', 0.005))
        # enforce portfolio-level cap
        max_risk_allowed = self.cfg.max_portfolio_risk - total_open_risk
        effective_risk = min(risk_per_trade, max(0.0, max_risk_allowed))
        risk_amt = equity * effective_risk

        sl_distance = self.cfg.atr_multiplier_sl * atr
        if sl_distance <= 0:
            logger.warning("SL distance <= 0, cannot compute lots")
            return 0.0

        units = risk_amt / (sl_distance * pip_value)
        lots = np.clip(units * pip_size, 0.01, 5.0)
        logger.info(f"Position sizing: equity={equity:.2f}, ATR={atr:.5f}, lots={lots:.2f}, effective_risk={effective_risk:.4f}")
        return round(lots, 2)

    # --- SL/TP targets ---
    def stop_targets(self, price: float, atr: float, direction: str, auc_score: float):
        sl_mult = self.cfg.atr_multiplier_sl
        tp_mult = self._get_dynamic_value(self.cfg.dynamic_tp, auc_score,
                                          getattr(self.cfg, 'atr_multiplier_tp', 2.5))
        if direction == "long":
            sl = price - sl_mult * atr
            tp = price + tp_mult * atr
        else:
            sl = price + sl_mult * atr
            tp = price - tp_mult * atr
        logger.debug(f"Stop targets: dir={direction}, price={price:.5f}, SL={sl:.5f}, TP={tp:.5f}")
        return sl, tp

    # --- Trading permission check ---
    def should_trade(self, now_local: pd.Timestamp, drawdown: float) -> bool:
        if drawdown >= self.cfg.block_on_drawdown:
            logger.info(f"Trading blocked: drawdown {drawdown:.3f} >= max {self.cfg.block_on_drawdown}")
            return False

        sess = self.cfg.session_filter
        if sess:
            start = pd.to_datetime(sess["start"]).time()
            end = pd.to_datetime(sess["end"]).time()
            allowed = start <= now_local.time() <= end
            if not allowed:
                logger.info(f"Trading blocked: outside session {start}-{end}, current={now_local.time()}")
            return allowed
        return True

    # --- Manage open positions: BE + ATR trailing ---
    def manage_open_positions(self, symbol: str, atr: float):
        import MetaTrader5 as mt5
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            logger.debug(f"[{symbol}] No open positions")
            return

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Symbol info unavailable")
            return

        point = symbol_info.point
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning(f"[{symbol}] Tick info unavailable")
            return

        for pos in positions:
            entry = pos.price_open
            sl = pos.sl
            direction = "long" if pos.type == mt5.ORDER_TYPE_BUY else "short"

            # Profit in pips
            profit_pips = (tick.bid - entry) / point if direction == "long" else (entry - tick.ask) / point
            one_r = self.cfg.atr_multiplier_sl * atr / point
            new_sl = sl

            # --- Breakeven at 1R ---
            if direction == "long" and profit_pips >= one_r and sl < entry:
                new_sl = entry
            elif direction == "short" and profit_pips >= one_r and sl > entry:
                new_sl = entry

            # --- ATR trailing ---
            trailing = atr * self.cfg.trailing_atr_mult
            if direction == "long":
                new_sl = max(new_sl, tick.bid - trailing)
            else:
                new_sl = min(new_sl, tick.ask + trailing)

            # Update SL if changed
            if new_sl != sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                result = mt5.order_send(request)
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"[{symbol}] Failed SL update for ticket {pos.ticket}: {result}")
                else:
                    logger.info(f"[{symbol}] Updated SL for ticket {pos.ticket}: new SL={new_sl:.5f}")
            else:
                logger.debug(f"[{symbol}] No SL adjustment needed for ticket {pos.ticket}")
