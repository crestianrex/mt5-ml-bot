# src/risk.py
from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from .config import Cfg
import datetime
from datetime import timezone, timedelta

class RiskManager:
    """
    RiskManager handles dynamic position sizing, SL/TP, portfolio exposure caps,
    open-position bookkeeping, and watchdog/cooldown behavior.

    Callers: pass `cfg` (the Cfg object) to constructor so both risk and watchdog settings are available.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.risk_cfg = cfg.risk
        self.watchdog_cfg = cfg.watchdog
        self.equity_peak: float | None = None
        self.open_positions_cache: dict[str, dict] = {}
        self.cooldown_until: datetime.datetime | None = None

    # ---------- Dynamic value helpers ----------
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

    # ---------- Position sizing ----------
    def position_size(self, equity: float, atr: float, pip_value: float, pip_size: float, auc_score: float, total_open_risk: float = 0.0) -> float:
        # dynamic risk fraction per trade
        risk_per_trade = self._get_dynamic_value(self.risk_cfg.dynamic_risk, auc_score, getattr(self.risk_cfg, "risk_per_trade", 0.005))
        max_risk_allowed = max(0.0, float(self.risk_cfg.max_portfolio_risk) - float(total_open_risk))
        effective_risk = min(risk_per_trade, max_risk_allowed)
        # absolute $ amount to risk
        risk_amt = float(equity) * float(effective_risk)

        sl_distance = float(self.risk_cfg.atr_multiplier_sl) * float(atr)
        if sl_distance <= 0 or (pip_value is None) or pip_value <= 0:
            logger.warning("Invalid SL distance or pip_value when computing position size")
            return 0.0

        if pip_size <= 0:
            logger.warning(f"Invalid pip_size for position sizing: {pip_size}")
            return 0.0
        
        sl_in_pips = sl_distance / pip_size
        risk_per_lot = sl_in_pips * pip_value

        if risk_per_lot <= 0:
            logger.warning(f"Calculated risk per lot is not positive: {risk_per_lot}")
            return 0.0

        units = risk_amt / risk_per_lot
        lots = float(np.clip(units, 0.01, 100.0))
        logger.info(f"Position sizing: equity={equity:.2f}, ATR={atr:.6f}, lots={lots:.4f}, effective_risk={effective_risk:.6f}")

        # round to 2 decimal lots (depends on broker; adjust if necessary)
        return round(lots, 2)

    # ---------- SL / TP ----------
    def stop_targets(self, price: float, atr: float, direction: str, auc_score: float, symbol: str):
        sl_mult = float(self.risk_cfg.atr_multiplier_sl)
        tp_mult = float(self._get_dynamic_value(self.risk_cfg.dynamic_tp, auc_score, float(self.risk_cfg.atr_multiplier_tp)))
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

    # ---------- Watchdog / cooldown helpers ----------
    def _update_equity_peak(self, equity_value: float):
        if self.equity_peak is None or equity_value > self.equity_peak:
            self.equity_peak = equity_value
            logger.debug(f"Equity peak updated: {self.equity_peak:.2f}")

    def _drawdown_exceeded(self, equity_value: float) -> bool:
        if self.equity_peak is None:
            return False
        dd = 1.0 - (equity_value / self.equity_peak) if self.equity_peak else 0.0
        if dd >= getattr(self.risk_cfg, "block_on_drawdown", 0.10):
            logger.warning(f"Drawdown threshold exceeded: drawdown={dd:.4f} >= {self.risk_cfg.block_on_drawdown}")
            return True
        return False

    def _count_consecutive_losses(self, lookback_hours: int = 48) -> int:
        import MetaTrader5 as mt5  # type: ignore
        """
        Query MT5 deal history in the last `lookback_hours` and compute the number
        of most recent consecutive losing closed trades (profit < 0).
        """
        try:
            now = datetime.datetime.now(timezone.utc)
            since = now - timedelta(hours=lookback_hours)
            # fetch recent deals
            deals = mt5.history_deals_get(since, now)
            if not deals:
                return 0
            # Convert to list sorted by time ascending
            recs = sorted(list(deals), key=lambda d: getattr(d, "time", 0))
            # get only deals with non-zero profit (closed)
            profits = []
            for d in recs:
                p = float(getattr(d, "profit", 0.0))
                # skip 0-profit deals (e.g., internal adjustments)
                if abs(p) > 1e-9:
                    profits.append(p)
            # count last consecutive negatives from end
            count = 0
            for p in reversed(profits):
                if p < 0:
                    count += 1
                else:
                    break
            logger.debug(f"Consecutive losing closed trades in last {lookback_hours}h: {count}")
            return count
        except Exception as e:
            logger.exception(f"_count_consecutive_losses failed: {e}")
            return 0

    def _trigger_cooldown(self, now: datetime.datetime | None = None):
        hours = float(getattr(self.watchdog_cfg, "cooldown_hours", 1.0))
        current_time = now if now is not None else datetime.datetime.now(timezone.utc)
        self.cooldown_until = current_time + timedelta(hours=hours)
        logger.warning(f"Watchdog triggered cooldown until {self.cooldown_until.isoformat()}")

    def cooldown_active(self, now: datetime.datetime | None = None) -> bool:
        if self.cooldown_until is None:
            return False
        
        current_time = now if now is not None else datetime.datetime.now(timezone.utc)

        if current_time < self.cooldown_until:
            return True
        
        # cooldown finished
        self.cooldown_until = None
        return False

    # ---------- Exposed check for trading permission ----------
    def should_trade(self, now_local: pd.Timestamp, drawdown: float) -> bool:
        import MetaTrader5 as mt5  # type: ignore
        """
        Returns True if trading is allowed.
        This function now enforces:
        - equity drawdown block (block_on_drawdown)
        - watchdog consecutive losses/cooldown
        - session filter
        """

        # 1) cooldown check (highest priority)
        if self.cooldown_active():
            logger.info(f"Trading blocked: watchdog cooldown active until {self.cooldown_until.isoformat()}")
            return False

        # 2) drawdown check (based on cfg.block_on_drawdown)
        # we'll also update equity peak if current equity available via mt5.account_info()
        try:
            acct = mt5.account_info()
            if acct:
                equity = float(getattr(acct, "equity", 0.0))
                self._update_equity_peak(equity)
                if self._drawdown_exceeded(equity):
                    # trigger cooldown
                    self._trigger_cooldown()
                    return False
        except Exception:
            logger.debug("should_trade: account_info() unavailable for drawdown checks")

        # 3) check watchdog: consecutive losses
        max_losses = getattr(self.watchdog_cfg, "max_consecutive_losses", None)
        if max_losses is not None and max_losses > 0:
            lost = self._count_consecutive_losses()
            if lost >= max_losses:
                logger.warning(f"Watchdog: consecutive losses {lost} >= threshold {max_losses}. Triggering cooldown.")
                self._trigger_cooldown()
                return False

        # 4) session filter
        sess = self.risk_cfg.session_filter
        if sess:
            try:
                start_t = pd.to_datetime(sess["start"]).time()
                end_t = pd.to_datetime(sess["end"]).time()
                allowed = start_t <= now_local.time() <= end_t
                if not allowed:
                    logger.info(f"Trading blocked: outside session {start_t}-{end_t}, current={now_local.time()}")
                    return False
            except Exception:
                logger.warning("Invalid session_filter in config; allowing trades by default.")
                return True

        # 5) block on drawdown parameter (if provided separately)
        if drawdown >= getattr(self.risk_cfg, "block_on_drawdown", 0.10):
            logger.info(f"Trading blocked: drawdown {drawdown:.3f} >= {self.risk_cfg.block_on_drawdown}")
            return False

        # allowed by default
        return True

    # ---------- Manage open positions (unchanged mostly) ----------
    def manage_open_positions(self, symbol: str, atr: float):
        import MetaTrader5 as mt5  # type: ignore
        """
        Ensure open_positions_cache matches MT5 positions and apply BE/trailing rules.
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

            one_r = (self.risk_cfg.atr_multiplier_sl * float(atr)) / point
            new_sl = sl

            # Breakeven at 1R
            if profit_pips >= one_r and ((direction == "long" and sl < entry) or (direction == "short" and sl > entry)):
                new_sl = entry

            # ATR trailing
            trailing = float(atr) * float(self.risk_cfg.trailing_atr_mult)
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
