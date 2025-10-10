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
from typing import List, Optional, Dict # Import Dict
# from backtester import SimPosition # Import SimPosition - REMOVED as not used
import datetime # NEW
from .notifier import TelegramNotifier # NEW import

@dataclass
class OrderResult:
    ok: bool
    ticket: int | None
    message: str

@dataclass
class ClosedTrade:
    """A comprehensive closed trade object for all post-trade processing."""
    ticket: int
    symbol: str
    direction: str
    lots: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    profit: float
    entry_equity: float
    risk_fraction: float
    exit_equity: float
    atr: float
    atr_idx: int
    min_prob_idx: int
    entry_auc: float
    adx: float = 0.0
    macd_diff: float = 0.0
    volatility_10: float = 0.0
    dist_from_ema_200: float = 0.0
    lagged_vol: float = 0.0 # NEW
    vol_drawdown_interaction: float = 0.0 # NEW

    def __repr__(self):
        return f"<ClosedTrade ticket={self.ticket}, profit={self.profit:.2f}>"

class Execution:
    """ Handles trade decision & order sending with retries + dry-run. """

    def __init__(self, ens_per_symbol: Dict[str, Ensemble], risk_manager: RiskManager, mt5_client, dry_run: bool = False, notifier: Optional[TelegramNotifier] = None):
        self.ens_per_symbol = ens_per_symbol # Changed from self.ens = ensemble
        self.risk = risk_manager
        self.mt5_client = mt5_client # NEW: Store MT5 client
        self.dry_run = dry_run
        self.notifier = notifier # NEW
        self._open_tickets = {}   # ticket -> dict of trade details from risk.open_positions_cache
        self._seen_closed = set() # to avoid reporting the same trade twice
        self._last_deal_time = 0  # NEW: Timestamp of the last deal processed

    def reconcile_open_positions_with_mt5(self):
        """
        Queries MT5 for all currently open positions and updates the internal
        open_positions_cache to reflect the ground truth from the broker.
        This is crucial for maintaining state across bot restarts.
        """
        logger.info("Reconciling open positions with MT5...")
        try:
            # 1. Get all open positions from MT5
            mt5_open_positions = mt5.positions_get() or []
            
            # 2. Clear the existing internal cache
            self.risk.open_positions_cache.clear()

            # 3. Populate the internal cache with positions from MT5
            for pos in mt5_open_positions:
                direction = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                entry_time_dt = datetime.datetime.fromtimestamp(pos.time, tz=datetime.timezone.utc)

                self.risk.open_positions_cache[pos.ticket] = {
                    "risk": 0.0, # Cannot infer from MT5 position directly
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "entry_price": pos.price_open,
                    "direction": direction,
                    "lots": pos.volume,
                    "entry_time": entry_time_dt,
                    "atr": 0.0, # Placeholder
                    "entry_auc": 0.5, # Placeholder
                    "risk_fraction": 0.0, # Placeholder
                    "entry_equity": 0.0, # Placeholder
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "atr_idx": -1, # Placeholder
                    "min_prob_idx": -1, # Placeholder
                    "adx": 0.0, # Placeholder
                    "macd_diff": 0.0, # Placeholder
                    "volatility_10": 0.0, # Placeholder
                    "dist_from_ema_200": 0.0, # Placeholder
                }
                logger.info(f"Reconciled open position: Ticket={pos.ticket}, Symbol={pos.symbol}, Direction={direction}, Lots={pos.volume}")
            
            logger.info(f"Reconciliation complete. {len(self.risk.open_positions_cache)} open positions tracked.")

        except Exception as e:
            logger.exception(f"Failed to reconcile open positions with MT5: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Failed to reconcile open positions with MT5: {e}", level="ERROR")

    def _send_order_with_retry(self, request: dict, retries: int = -1, delay: float = 1.0):
        num_retries = self.risk.cfg.trading_costs.defaults.retry_order_send if retries == -1 else retries
        last = None
        for attempt in range(1, num_retries + 1):
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

    def check_closed_trades(self) -> List[ClosedTrade]:
        """
        Reconciles the internal cache of open positions with the broker's state.
        Returns a list of newly detected closed trades.
        """
        closed_trades_list = []
        try:
            # Get the ground truth of open positions from the broker
            open_positions_on_broker = mt5.positions_get() or []
            open_position_ids_on_broker = {pos.ticket for pos in open_positions_on_broker}

            # Get the list of positions we are tracking internally
            tracked_position_ids = list(self.risk.open_positions_cache.keys())

            # Find positions that are in our cache but not in the broker's list of open positions
            closed_pids = [pid for pid in tracked_position_ids if pid not in open_position_ids_on_broker]

            for pid in closed_pids:
                trade_details = self.risk.open_positions_cache.get(pid)
                if not trade_details:
                    continue

                # Fetch the deal history for this specific closed position to find the PnL
                deals = mt5.history_deals_get(position=pid)
                if not deals:
                    logger.warning(f"Position {pid} is closed but no deal history found. Removing from cache.")
                    self.risk.open_positions_cache.pop(pid, None)
                    continue

                # Find the closing deal to get the final profit and exit details
                final_profit = 0.0
                last_exit_time = None
                last_exit_price = None
                for deal in sorted(deals, key=lambda d: d.time):
                    if deal.entry == mt5.DEAL_ENTRY_OUT:
                        final_profit += deal.profit
                        last_exit_time = deal.time
                        last_exit_price = deal.price

                if last_exit_time is None:
                    logger.warning(f"Position {pid} is closed but no 'out' deal found. Removing from cache.")
                    self.risk.open_positions_cache.pop(pid, None)
                    continue

                # Get current equity for the ClosedTrade object
                account_info = mt5.account_info()
                actual_equity = getattr(account_info, "equity", 0.0) if account_info else 0.0
                exit_time_dt = datetime.datetime.fromtimestamp(last_exit_time, tz=datetime.timezone.utc)

                # Create the comprehensive ClosedTrade object
                closed_trade = ClosedTrade(
                    ticket=pid,
                    symbol=trade_details.get("symbol", "UNKNOWN"),
                    direction=trade_details.get("direction", ""),
                    lots=trade_details.get("lots", 0.0),
                    entry_price=trade_details.get("entry_price", 0.0),
                    exit_price=last_exit_price or 0.0,
                    entry_time=trade_details.get("entry_time"),
                    exit_time=exit_time_dt,
                    profit=final_profit,
                    entry_equity=trade_details.get("entry_equity", 0.0),
                    risk_fraction=trade_details.get("risk_fraction", 0.0),
                    exit_equity=actual_equity,
                    atr=trade_details.get("atr", 0.0),
                    atr_idx=trade_details.get("atr_idx", -1),
                    min_prob_idx=trade_details.get("min_prob_idx", -1),
                    entry_auc=trade_details.get("entry_auc", 0.5),
                    adx=trade_details.get("adx", 0.0),
                    macd_diff=trade_details.get("macd_diff", 0.0),
                    volatility_10=trade_details.get("volatility_10", 0.0),
                    dist_from_ema_200=trade_details.get("dist_from_ema_200", 0.0),
                    lagged_vol=trade_details.get("lagged_vol", 0.0), # NEW
                    vol_drawdown_interaction=trade_details.get("vol_drawdown_interaction", 0.0) # NEW
                )
                closed_trades_list.append(closed_trade)
                logger.info(f"Detected closed trade via reconciliation: {closed_trade}")

                # Remove the now-closed position from our internal cache
                self.risk.open_positions_cache.pop(pid, None)

        except Exception as e:
            logger.exception(f"Failed to check/reconcile closed trades: {e}")
        
        return closed_trades_list

    def _manage_trailing_stops(self):
        if not self.risk.cfg.risk.trailing_stop.get("enabled", False):
            return

        atr_distance = self.risk.cfg.risk.trailing_stop.get("atr_distance", 1.5)

        for ticket, position in list(self.risk.open_positions_cache.items()):
            symbol = position.get("symbol")
            if not symbol:
                continue

            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                continue

            current_price = tick.bid if position["direction"] == "long" else tick.ask
            entry_price = position["entry_price"]
            current_sl = position.get("sl", 0.0)
            atr = position.get("atr", 0.0)

            if atr <= 0:
                continue

            new_sl = 0.0
            if position["direction"] == "long" and current_price > entry_price:
                new_sl = current_price - atr * atr_distance
                if new_sl > current_sl:
                    self._modify_position(ticket, new_sl, position.get("tp"))

            elif position["direction"] == "short" and current_price < entry_price:
                new_sl = current_price + atr * atr_distance
                if new_sl < current_sl:
                    self._modify_position(ticket, new_sl, position.get("tp"))

    def _modify_position(self, ticket, sl, tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(sl),
            "tp": float(tp),
        }
        if self.dry_run:
            logger.info(f"[DRY-RUN] Modifying position {ticket}: SL={sl}, TP={tp}")
            return

        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
            # Update local cache
            if ticket in self.risk.open_positions_cache:
                self.risk.open_positions_cache[ticket]["sl"] = sl
        else:
            logger.error(f"Failed to modify position {ticket}: {res}")

    def trade(self, symbol: str, X: pd.DataFrame | None = None, atr: float | None = None, auc_score: float | None = 0.5, total_open_risk: float = 0.0, sl_mult: Optional[float] = None, tp_mult: Optional[float] = None, atr_idx: int = -1, min_prob_idx: int = -1, lagged_vol: float = 0.0, vol_drawdown_interaction: float = 0.0) -> OrderResult:
        if X is None or atr is None:
            return OrderResult(False, None, "X or ATR missing")

        # Predict
        try:
            ens = self.ens_per_symbol.get(symbol) # Get the specific ensemble for this symbol
            if ens is None:
                error_msg = f"<b>ERROR:</b> No ensemble found for symbol {symbol}."
                logger.error(error_msg)
                if self.notifier: self.notifier.send_message(error_msg, level="ERROR")
                return OrderResult(False, None, error_msg)

            prob_series = ens.predict_proba(X.iloc[[-1]])
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
        if ens.best_threshold_ is not None:
            threshold = ens.best_threshold_
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

        pip_value = None
        # 1. Check for symbol override in config
        if symbol in self.risk.cfg.symbol_overrides and "pip_value" in self.risk.cfg.symbol_overrides[symbol]:
            pip_value = self.risk.cfg.symbol_overrides[symbol]["pip_value"]
            logger.info(f"[{symbol}] Using configured override pip_value: {pip_value}")

        # 2. If no override, try to calculate it
        if pip_value is None:
            pip_size = getattr(symbol_info, "point", None)
            contract_size = getattr(symbol_info, "trade_contract_size", 1.0)
            if pip_size and contract_size:
                pip_value = pip_size * contract_size
        
        # 3. Final check and fail-safe
        if pip_value is None or pip_value <= 0:
            error_msg = f"CRITICAL: Could not determine pip_value for {symbol}. Cannot calculate position size. Please set it in config.yaml under symbol_overrides."
            logger.critical(error_msg)
            if self.notifier: self.notifier.send_message(error_msg, level="CRITICAL")
            return OrderResult(False, None, "Invalid pip_value")

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Tick info unavailable for {symbol}.", level="ERROR")
            return OrderResult(False, None, "Tick info unavailable")
        spread_value = float(tick.ask) - float(tick.bid)

        lots = self.risk.position_size(equity, atr, pip_value, pip_size, auc_score, spread_value, total_open_risk)
        if lots <= 0:
            return OrderResult(False, None, "Lots <= 0")

        price = float(tick.ask) if direction == "long" else float(tick.bid)

        sl, tp = self.risk.stop_targets(price=price, atr=atr, direction=direction, auc_score=auc_score, symbol=symbol, sl_mult=sl_mult, tp_mult=tp_mult)
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
            "magic": self.risk.cfg.magic_number,
            "comment": "ml-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if self.dry_run:
            logger.info(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}")
            if self.notifier: self.notifier.send_message(f"[DRY-RUN] Prepared {direction} for {symbol}: lots={lots}, SL={sl}, TP={tp}", level="INFO")
            return OrderResult(True, None, "Dry-run prepared")

        logger.debug(f"[{symbol}] Sending order request: {request}")
        res = self._send_order_with_retry(request)
        if res is None or getattr(res, "retcode", None) != getattr(mt5, "TRADE_RETCODE_DONE", 10009):
            error_msg = f"<b>CRITICAL:</b> Order failed for {symbol} after retries: {res}"
            logger.error(error_msg)
            if self.notifier: self.notifier.send_message(error_msg, level="CRITICAL")
            return OrderResult(False, getattr(res, "order", None) if res else None, f"Order failed: {res}")

        deal_ticket = getattr(res, "deal", None)
        if not deal_ticket:
            logger.error(f"Order for {symbol} succeeded but no deal ticket returned. Cannot track position.")
            return OrderResult(False, None, "Order sent but no deal ticket.")

        # Fetch the deal to get the position_id, which is the reliable key
        deals = mt5.history_deals_get(ticket=deal_ticket)
        if not deals:
            logger.error(f"Could not fetch deal info for deal {deal_ticket}. Cannot track position.")
            return OrderResult(False, None, "Failed to fetch deal info.")
        
        position_id = deals[0].position_id

        logger.info(f"Order executed: ticket={getattr(res, 'order', None)}, position_id={position_id}, dir={direction}, lots={lots}, SL={sl}, TP={tp}")
        if self.notifier: self.notifier.send_message(f"<b>TRADE EXECUTED:</b> {direction} {lots} lots of {symbol} at {price:.5f}. SL:{sl:.5f} TP:{tp:.5f}", level="INFO")

        # compute effective risk and store in cache keyed by the reliable position_id
        try:
            risk_per_trade = self.risk._get_dynamic_value(self.risk.risk_cfg.dynamic_risk, auc_score, getattr(self.risk.risk_cfg, "risk_per_trade", 0.005))
            risk_amt = equity * risk_per_trade
            sl_distance = max(1e-6, self.risk.risk_cfg.atr_multiplier_sl * atr)
            effective_lots = (risk_amt / (sl_distance * pip_value)) if pip_value and sl_distance else 0.0
            
            # Store comprehensive details for later SimPosition reconstruction
            self.risk.open_positions_cache[position_id] = { # Use position_id as key
                "risk": float(risk_amt), # Store the dollar amount at risk
                "ticket": position_id, # Store position_id for consistency
                "symbol": symbol, # NEW: Store symbol
                "entry_price": price,
                "direction": direction,
                "lots": float(lots),
                "entry_time": datetime.datetime.now(datetime.timezone.utc), # Use current UTC time
                "atr": atr, # ATR at the time of entry
                "entry_auc": auc_score, # AUC at the time of entry
                "risk_fraction": risk_per_trade, # Store the risk_per_trade as risk_fraction
                "entry_equity": equity, # Store equity at the time of entry
                "sl": sl, # SL at entry
                "tp": tp, # TP at entry
                "atr_idx": atr_idx,
                "min_prob_idx": min_prob_idx,
                "adx": float(X["adx"].iloc[-1]) if "adx" in X.columns else 0.0,
                "macd_diff": float(X["macd_diff"].iloc[-1]) if "macd_diff" in X.columns else 0.0,
                "volatility_10": float(X["volatility_10"].iloc[-1]) if "volatility_10" in X.columns else 0.0,
                "dist_from_ema_200": float(X["dist_from_ema_200"].iloc[-1]) if "dist_from_ema_200" in X.columns else 0.0,
                # NEW: Add inter_market_feature and mta_feature to open_positions_cache
                "inter_market_feature": float(X["inter_market_feature"].iloc[-1]) if "inter_market_feature" in X.columns else 0.0,
                "mta_feature": float(X["mta_feature"].iloc[-1]) if "mta_feature" in X.columns else 0.0,
                "lagged_vol": lagged_vol, # NEW
                "vol_drawdown_interaction": vol_drawdown_interaction # NEW
            }
        except Exception as e:
            logger.warning(f"Could not record open position in cache: {e}")
            if self.notifier: self.notifier.send_message(f"<b>WARNING:</b> Could not record open position in cache for {symbol}: {e}", level="WARNING")

        return OrderResult(True, position_id, "OK")
