# src/risk.py
from __future__ import annotations
import pandas as pd
import numpy as np
from loguru import logger
from .config import Cfg
import datetime
from datetime import timezone, timedelta
from typing import List, Optional # Import Optional
from .trade import SimPosition # Import SimPosition
from .notifier import TelegramNotifier # NEW import

class RiskManager:
    """
    RiskManager handles dynamic position sizing, SL/TP, portfolio exposure caps,
    open-position bookkeeping, and watchdog/cooldown behavior.

    Callers: pass `cfg` (the Cfg object) to constructor so both risk and watchdog settings are available.
    """

    def __init__(self, cfg: Cfg, notifier: Optional[TelegramNotifier] = None):
        self.cfg = cfg
        self.risk_cfg = cfg.risk
        self.watchdog_cfg = cfg.watchdog
        self.equity_peak: float | None = None
        self.open_positions_cache: dict[str, dict] = {}
        self.cooldown_until: datetime.datetime | None = None
        self.recently_closed_trades: List[SimPosition] = [] # New: To store closed trades for monitoring
        self.notifier = notifier # NEW

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
    def stop_targets(self, price: float, atr: float, direction: str, auc_score: float, symbol: str, sl_mult: float | None = None, tp_mult: float | float | None = None):
        _sl_mult = sl_mult if sl_mult is not None else float(self.risk_cfg.atr_multiplier_sl)
        _tp_mult = tp_mult if tp_mult is not None else float(self._get_dynamic_value(self.risk_cfg.dynamic_tp, auc_score, float(self.risk_cfg.atr_multiplier_tp)))
        price = float(price)
        atr = float(atr)
        if direction == "long":
            sl = price - _sl_mult * atr
            tp = price + _tp_mult * atr
        else:
            sl = price + _sl_mult * atr
            tp = price - _tp_mult * atr
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
            logger.warning(f"Drawdown threshold exceeded: equity={equity_value:.2f}, peak={self.equity_peak:.2f}, drawdown={dd:.4f} >= {self.risk_cfg.block_on_drawdown}")
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
        message = f"<b>RISK ALERT:</b> Watchdog triggered cooldown until {self.cooldown_until.isoformat()}"
        logger.warning(message)
        if self.notifier: self.notifier.send_message(message, level="WARNING")

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
                message = f"<b>RISK ALERT:</b> Watchdog: consecutive losses {lost} >= threshold {max_losses}. Triggering cooldown."
                logger.warning(message)
                if self.notifier: self.notifier.send_message(message, level="WARNING")
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
            message = f"<b>RISK ALERT:</b> Trading blocked: drawdown {drawdown:.3f} >= {self.risk_cfg.block_on_drawdown}"
            logger.info(message)
            if self.notifier: self.notifier.send_message(message, level="INFO")
            return False

        # allowed by default
        return True

    # ---------- Manage open positions (unchanged mostly) ----------
    def manage_open_positions(self, symbol: str, atr: float) -> List[SimPosition]:
        import MetaTrader5 as mt5  # type: ignore
        """
        Ensure open_positions_cache matches MT5 positions and apply BE/trailing rules.
        Returns a list of SimPosition objects for trades that were closed in this cycle.
        """
        self.recently_closed_trades.clear() # Clear for this cycle

        try:
            mt5_positions = mt5.positions_get(symbol=symbol) or []
            current_tickets = {int(p.ticket) for p in mt5_positions}
        except Exception as e:
            logger.exception(f"Failed to get MT5 positions for {symbol}: {e}")
            return []

        # Identify and process closed positions
        closed_tickets_in_cache = []
        for sym_key, pos_data in list(self.open_positions_cache.items()):
            ticket = pos_data.get("ticket")
            if ticket is not None and ticket not in current_tickets:
                closed_tickets_in_cache.append(ticket)
                # Remove from cache
                del self.open_positions_cache[sym_key]
                logger.info(f"[{symbol}] Removed closed position {ticket} from cache.")

                # Fetch deal history to reconstruct SimPosition
                deals = mt5.history_deals_get(position=ticket)
                if deals:
                    # Find the deal that closed the position (usually the last one)
                    closing_deal = None
                    for deal in deals:
                        # Check for actual trade deals (buy/sell) that have profit/loss
                        if (deal.type == mt5.DEAL_TYPE_BUY or deal.type == mt5.DEAL_TYPE_SELL) and abs(deal.profit) > 1e-9:
                            closing_deal = deal
                    
                    if closing_deal:
                        # Reconstruct SimPosition from deal and cached info
                        entry_price = pos_data.get("entry_price", 0.0)
                        direction = pos_data.get("direction", "long")
                        lots = pos_data.get("lots", 0.0)
                        entry_time = pos_data.get("entry_time", datetime.datetime.now(timezone.utc))
                        sl = pos_data.get("sl", 0.0)
                        tp = pos_data.get("tp", 0.0)
                        entry_auc = pos_data.get("entry_auc", 0.5)
                        risk_fraction = pos_data.get("risk_fraction", 0.0)
                        atr_at_entry = pos_data.get("atr", 0.0)

                        pnl = float(closing_deal.profit)
                        exit_price = float(closing_deal.price)
                        exit_time = datetime.datetime.fromtimestamp(closing_deal.time, tz=timezone.utc)

                        closed_sim_pos = SimPosition(
                            symbol=symbol,
                            direction=direction,
                            lots=lots,
                            entry_price=entry_price,
                            sl=sl,
                            tp=tp,
                            entry_time=entry_time,
                            atr=atr_at_entry,
                            entry_auc=entry_auc,
                            risk_fraction=risk_fraction
                        )
                        closed_sim_pos.close(exit_price, exit_time, pnl)
                        self.recently_closed_trades.append(closed_sim_pos)
                        logger.info(f"[{symbol}] Detected closed trade {ticket}. PnL: {pnl:.2f}")
                    else:
                        logger.warning(f"[{symbol}] Could not find a valid closing deal for ticket {ticket}.")
                else:
                    logger.warning(f"[{symbol}] No deal history found for closed position {ticket}.")

        if not mt5_positions:
            logger.debug(f"[{symbol}] No open positions in MT5.")
            return self.recently_closed_trades

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] Symbol info unavailable")
            return self.recently_closed_trades
        point = float(symbol_info.point)
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning(f"[{symbol}] Tick info unavailable")
            return self.recently_closed_trades

        for pos in mt5_positions:
            entry = float(pos.price_open)
            sl = float(getattr(pos, "sl", 0.0))
            tp = float(getattr(pos, "tp", 0.0))
            direction = "long" if int(pos.type) == int(mt5.ORDER_TYPE_BUY) else "short"

            # Update open_positions_cache with more details for later reconstruction
            # Ensure existing details are preserved if not updated here
            cached_pos_data = self.open_positions_cache.get(symbol, {})
            self.open_positions_cache[symbol] = {
                "risk": cached_pos_data.get("risk", 0.0), # Keep existing risk
                "ticket": int(pos.ticket),
                "entry_price": entry,
                "direction": direction,
                "lots": float(pos.volume),
                "entry_time": datetime.datetime.fromtimestamp(pos.time_setup, tz=timezone.utc),
                "sl": sl,
                "tp": tp,
                "atr": cached_pos_data.get("atr", 0.0), # Preserve ATR from entry
                "entry_auc": cached_pos_data.get("entry_auc", 0.5), # Preserve AUC from entry
                "risk_fraction": cached_pos_data.get("risk_fraction", 0.0), # Preserve risk_fraction from entry
            }

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
        
        return self.recently_closed_trades
