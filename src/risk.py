# src/risk.py
from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from .config import RiskCfg
import MetaTrader5 as mt5  # type: ignore


class RiskManager:
    """
    RiskManager handles dynamic position sizing, SL/TP, portfolio exposure caps and open-position bookkeeping.
    open_positions_cache: { symbol: {"risk": float, "ticket": int} }
    """

    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg
        self.equity_peak = None
        self.open_positions_cache: dict[str, dict] = {}

    def _get_dynamic_value(self, dynamic_cfg: dict | None, auc_score: float, default_val: float) -> float:
        if not dynamic_cfg or not dynamic_cfg.get("enabled"):
            return float(default_val)
        auc_floor = float(dynamic_cfg.get("auc_floor", 0.55))
        auc_ceiling = float(dynamic_cfg.get("auc_ceiling", 0.65))
        base_val = float(dynamic_cfg.get("base_risk", dynamic_cfg.get("base_tp_mult", default_val)))
        max_val = float(dynamic_cfg.get("max_risk", dynamic_cfg.get("max_tp_mult", default_val)))
        clamped = float(np.clip(auc_score, auc_floor, auc_ceiling))
        denom = max(auc_ceiling - auc_floor, 1e-6)
        val = base_val + (clamped - auc_floor) * (max_val - base_val) / denom
        logger.debug(f"Dynamic value calc: AUC={auc_score:.4f}, Clamped={clamped:.4f}, Value={val:.4f}")
        return float(val)

    def position_size(self, equity: float, atr: float, pip_value: float, pip_size: float, auc_score: float, total_open_risk: float = 0.0) -> float:
        risk_per_trade = self._get_dynamic_value(self.cfg.dynamic_risk, auc_score, getattr(self.cfg, "risk_per_trade", 0.005))
        max_risk_allowed = max(0.0, self.cfg.max_portfolio_risk - float(total_open_risk))
        effective_risk = min(risk_per_trade, max_risk_allowed)
        risk_amt = float(equity) * float(effective_risk)
        sl_distance = float(self.cfg.atr_multiplier_sl) * float(atr)
        if sl_distance <= 0 or pip_value <= 0:
            logger.warning("Invalid SL distance or pip_value when computing position size")
            return 0.0
        units = risk_amt / (sl_distance * pip_value)
        lots = float(np.clip(units, 0.01, 100.0))
        logger.info(f"Position sizing: equity={equity:.2f}, ATR={atr:.6f}, lots={lots:.4f}, effective_risk={effective_risk:.6f}")
        # round to 2 decimal lots (depends on broker; adjust if necessary)
        return round(lots, 2)

    def stop_targets(self, price: float, atr: float, direction: str, auc_score: float, symbol: str):
        sl_mult = float(self.cfg.atr_multiplier_sl)
        tp_mult = float(self._get_dynamic_value(self.cfg.dynamic_tp, auc_score, float(self.cfg.atr_multiplier_tp)))
        price = float(price)
        atr = float(atr)
        if direction == "long":
            sl = price - sl_mult * atr
            tp = price + tp_mult * atr
        else:
            sl = price + sl_mult * atr
            tp = price - tp_mult * atr
        logger.debug(f"Stop targets: dir={direction}, price={price:.6f}, SL={sl:.6f}, TP={tp:.6f}")
        return float(sl), float(tp)

    def should_trade(self, now_local: pd.Timestamp, drawdown: float) -> bool:
        if drawdown >= self.cfg.block_on_drawdown:
            logger.info(f"Trading blocked: drawdown {drawdown:.3f} >= {self.cfg.block_on_drawdown}")
            return False
        sess = self.cfg.session_filter
        if sess:
            try:
                start_t = pd.to_datetime(sess["start"]).time()
                end_t = pd.to_datetime(sess["end"]).time()
                allowed = start_t <= now_local.time() <= end_t
                if not allowed:
                    logger.info(f"Trading blocked: outside session {start_t}-{end_t}, current={now_local.time()}")
                return allowed
            except Exception:
                logger.warning("Invalid session_filter in config; allowing trades by default.")
                return True
        return True

    def manage_open_positions(self, symbol: str, atr: float):
        """
        Ensure open_positions_cache matches MT5 positions and apply BE/trailing rules.
        open_positions_cache keys are symbols.
        """
        try:
            mt5_positions = mt5.positions_get(symbol=symbol) or []
            current_tickets = {int(p.ticket) for p in mt5_positions}
        except Exception as e:
            logger.exception(f"Failed to get MT5 positions for {symbol}: {e}")
            return

        # remove cache entries for which ticket is not present anymore
        symbols_to_remove = []
        for sym, pos_data in list(self.open_positions_cache.items()):
            ticket = pos_data.get("ticket")
            if ticket is None or ticket not in current_tickets:
                symbols_to_remove.append(sym)
        for sym in symbols_to_remove:
            try:
                del self.open_positions_cache[sym]
                logger.info(f"[{sym}] Removed closed position from cache.")
            except KeyError:
                pass

        if not mt5_positions:
            logger.debug(f"[{symbol}] No open positions in MT5.")
            return

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Symbol info unavailable")
            return

        point = float(symbol_info.point)
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning(f"[{symbol}] Tick info unavailable")
            return

        for pos in mt5_positions:
            entry = float(pos.price_open)
            sl = float(getattr(pos, "sl", 0.0))
            tp = float(getattr(pos, "tp", 0.0))
            direction = "long" if int(pos.type) == int(mt5.ORDER_TYPE_BUY) else "short"
            # profit in pips
            if direction == "long":
                profit_pips = (float(tick.bid) - entry) / point
            else:
                profit_pips = (entry - float(tick.ask)) / point

            one_r = (self.cfg.atr_multiplier_sl * float(atr)) / point
            new_sl = sl

            # Breakeven at 1R
            if profit_pips >= one_r and ((direction == "long" and sl < entry) or (direction == "short" and sl > entry)):
                new_sl = entry

            # ATR trailing
            trailing = float(atr) * float(self.cfg.trailing_atr_mult)
            if direction == "long":
                new_sl = max(new_sl, float(tick.bid) - trailing)
            else:
                new_sl = min(new_sl, float(tick.ask) + trailing)

            if abs(new_sl - sl) > 1e-8:
                try:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": int(pos.ticket),
                        "sl": float(new_sl),
                        "tp": float(tp),
                    }
                    result = mt5.order_send(request)
                    if result is None or getattr(result, "retcode", None) != getattr(mt5, "TRADE_RETCODE_DONE", 10009):
                        logger.error(f"[{symbol}] Failed SL update for ticket {pos.ticket}: {result}")
                    else:
                        logger.info(f"[{symbol}] Updated SL for ticket {pos.ticket}: new SL={new_sl:.6f}")
                except Exception as e:
                    logger.exception(f"[{symbol}] Exception updating SL for ticket {pos.ticket}: {e}")
            else:
                logger.debug(f"[{symbol}] No SL adjustment needed for ticket {pos.ticket}")
