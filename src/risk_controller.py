# src/risk_controller.py
import numpy as np
import pandas as pd
from loguru import logger
import datetime
from collections import deque
import json
import os
from typing import List, Dict, Tuple, Optional, Any

from src.config import Cfg
from src.trade import SimPosition # For reward normalization
from src.linear_thompson import LinearThompson  # new

def _json_serial(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class ThompsonBandit:
    """
    Normal/Gaussian Thompson bandit with empirical-per-arm variance estimation.
    This remains backwards-compatible with your previous usage.
    """
    def __init__(self, num_arms: int, prior_mean: float, prior_var: float, min_var: float = 1e-6):
        self.num_arms = int(num_arms)
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)
        self.min_var = float(min_var)

        # Sufficient statistics
        self.counts = np.zeros(self.num_arms, dtype=float)
        self.sum_rewards = np.zeros(self.num_arms, dtype=float)
        self.sum_squared_rewards = np.zeros(self.num_arms, dtype=float)

    def _emp_mean_var(self, i: int):
        n = self.counts[i]
        if n <= 0:
            return self.prior_mean, self.prior_var
        mean = self.sum_rewards[i] / n
        if n > 1:
            var = max(self.min_var, (self.sum_squared_rewards[i] - n * mean * mean) / (n - 1))
        else:
            var = max(self.min_var, self.prior_var)
        return float(mean), float(var)

    def sample(self) -> int:
        posterior_means = np.zeros(self.num_arms)
        posterior_vars = np.zeros(self.num_arms)

        for i in range(self.num_arms):
            mean_i, emp_var = self._emp_mean_var(i)
            denom = (1.0 / self.prior_var) + (self.counts[i] / emp_var if emp_var > 0 else 0.0)
            post_var = 1.0 / denom if denom > 0 else self.prior_var
            post_mean = (self.prior_mean / self.prior_var + (self.sum_rewards[i] / emp_var if emp_var > 0 else 0.0)) * post_var
            posterior_means[i] = post_mean
            posterior_vars[i] = max(post_var, 1e-12)

        samples = np.random.normal(posterior_means, np.sqrt(posterior_vars))
        return int(np.argmax(samples))

    def update(self, arm_index: int, reward: float, decay: float = 1.0):
        if decay != 1.0:
            self.counts *= decay
            self.sum_rewards *= decay
            self.sum_squared_rewards *= decay

        self.counts[arm_index] += 1.0
        self.sum_rewards[arm_index] += reward
        self.sum_squared_rewards[arm_index] += reward * reward

    def get_state(self):
        return {
            "num_arms": self.num_arms,
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "min_var": self.min_var,
            "counts": self.counts.tolist(),
            "sum_rewards": self.sum_rewards.tolist(),
            "sum_squared_rewards": self.sum_squared_rewards.tolist(),
        }

    @classmethod
    def from_state(cls, state):
        inst = cls(state["num_arms"], state.get("prior_mean", 0.0), state.get("prior_var", 1.0), state.get("min_var", 1e-6))
        inst.counts = np.array(state.get("counts", inst.counts))
        inst.sum_rewards = np.array(state.get("sum_rewards", inst.sum_rewards))
        inst.sum_squared_rewards = np.array(state.get("sum_squared_rewards", inst.sum_squared_rewards))
        return inst

class SymbolRiskState:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        ts_cfg = cfg.thompson_sampling

        # Store current dynamic grid values
        self.atr_grid_values: List[float] = list(ts_cfg.atr_grid)
        self.min_prob_grid_values: List[float] = list(ts_cfg.min_prob_grid)

        self.atr_bandit = ThompsonBandit(
            num_arms=len(self.atr_grid_values),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        self.min_prob_bandit = ThompsonBandit(
            num_arms=len(self.min_prob_grid_values),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        # optional contextual bandit (for ATR choices)
        self.contextual_bandit = None
        if getattr(ts_cfg, "contextual_enabled", False):
            # small default context dimension; RiskController will define how to build the vector
            ctx_dim = int(getattr(ts_cfg, "context_dim", 5))
            self.contextual_bandit = LinearThompson(
                num_arms=len(self.atr_grid_values),
                dim=ctx_dim,
                lambda_prior=1.0,
                noise_var=float(ts_cfg.obs_var or 1.0),
                dynamic_noise_var_enabled=ts_cfg.dynamic_noise_var_enabled,
                noise_var_window_size=ts_cfg.noise_var_window_size,
                min_noise_var=ts_cfg.min_noise_var,
                dynamic_uncertainty_risk_scaling_enabled=ts_cfg.dynamic_uncertainty_risk_scaling_enabled,
                uncertainty_risk_factor=ts_cfg.uncertainty_risk_factor,
                uncertainty_threshold=ts_cfg.uncertainty_threshold
            )

        self.peak_equity: float = cfg.initial_equity
        self.current_equity: float = cfg.initial_equity
        self.consecutive_losses: int = 0
        self.recent_returns: deque[float] = deque(maxlen=ts_cfg.rule_rolling_window)
        self.last_atr: float = 0.0

        # History for lagged features
        self.vol_history: deque[float] = deque(maxlen=ts_cfg.lagged_vol_period + 1) if ts_cfg.enable_lagged_vol else deque(maxlen=1)

        # Track updates for adaptive grids
        self.atr_updates_since_last_adaptation: int = 0
        self.min_prob_updates_since_last_adaptation: int = 0
        self.last_reset_time: Optional[datetime.datetime] = None # NEW: To track last reset for cooldown

    def _get_bandit_and_grid(self, param_type: str) -> Tuple[ThompsonBandit | LinearThompson, List[float]]:
        if param_type == "atr":
            if self.contextual_bandit is not None:
                return self.contextual_bandit, self.atr_grid_values
            return self.atr_bandit, self.atr_grid_values
        elif param_type == "min_prob":
            return self.min_prob_bandit, self.min_prob_grid_values
        raise ValueError(f"Unknown param_type: {param_type}")

    def get_state(self):
        d = {
            "atr_bandit": self.atr_bandit.get_state(),
            "min_prob_bandit": self.min_prob_bandit.get_state(),
            "atr_grid_values": self.atr_grid_values,
            "min_prob_grid_values": self.min_prob_grid_values,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "consecutive_losses": self.consecutive_losses,
            "recent_returns": list(self.recent_returns),
            "last_atr": self.last_atr,
            "last_reset_time": self.last_reset_time.isoformat() if self.last_reset_time else None,
            "vol_history": list(self.vol_history) if self.vol_history else [], # NEW
        }
        if self.contextual_bandit is not None:
            d["contextual_bandit"] = self.contextual_bandit.get_state()
        return d

    @classmethod
    def from_state(cls, cfg: Cfg, state: dict) -> "SymbolRiskState":
        inst = cls(cfg)
        inst.atr_grid_values = state.get("atr_grid_values", list(cfg.thompson_sampling.atr_grid))
        inst.min_prob_grid_values = state.get("min_prob_grid_values", list(cfg.thompson_sampling.min_prob_grid))
        # Re-initialize bandits with loaded grid sizes
        inst.atr_bandit = ThompsonBandit.from_state(state["atr_bandit"])
        inst.min_prob_bandit = ThompsonBandit.from_state(state["min_prob_bandit"])
        inst.peak_equity = state.get("peak_equity", inst.peak_equity)
        inst.current_equity = state.get("current_equity", inst.current_equity)
        inst.consecutive_losses = state.get("consecutive_losses", inst.consecutive_losses)
        inst.recent_returns = deque(state.get("recent_returns", []), maxlen=cfg.thompson_sampling.rule_rolling_window)
        inst.last_atr = state.get("last_atr", inst.last_atr)
        last_reset_time_str = state.get("last_reset_time")
        inst.last_reset_time = datetime.datetime.fromisoformat(last_reset_time_str) if last_reset_time_str else None
        # Load vol_history
        if cfg.thompson_sampling.enable_lagged_vol:
            inst.vol_history = deque(state.get("vol_history", []), maxlen=cfg.thompson_sampling.lagged_vol_period + 1)
        else:
            inst.vol_history = deque(maxlen=1)
        if "contextual_bandit" in state and getattr(cfg.thompson_sampling, "contextual_enabled", False):
            # Re-initialize contextual bandit with loaded grid size
            ctx_dim = int(getattr(cfg.thompson_sampling, "context_dim", 9))
            inst.contextual_bandit = LinearThompson(
                num_arms=len(inst.atr_grid_values),
                dim=ctx_dim,
                lambda_prior=1.0,
                noise_var=float(cfg.thompson_sampling.obs_var or 1.0),
                dynamic_noise_var_enabled=cfg.thompson_sampling.dynamic_noise_var_enabled,
                noise_var_window_size=cfg.thompson_sampling.noise_var_window_size,
                min_noise_var=cfg.thompson_sampling.min_noise_var,
                dynamic_uncertainty_risk_scaling_enabled=cfg.thompson_sampling.dynamic_uncertainty_risk_scaling_enabled,
                uncertainty_risk_factor=cfg.thompson_sampling.uncertainty_risk_factor,
                uncertainty_threshold=cfg.thompson_sampling.uncertainty_threshold
            )
            inst.contextual_bandit = LinearThompson.from_state(state["contextual_bandit"])
        return inst

class RiskController:
    """
    Manages Thompson Sampling bandits and rule-based scaling for multiple symbols.
    """
    def __init__(self, cfg: Cfg, notifier=None):
        self.cfg = cfg
        self.notifier = notifier
        self.symbol_states: Dict[str, SymbolRiskState] = {}
        for sym in cfg.symbols:
            self.symbol_states[sym] = SymbolRiskState(cfg)
        
        self.state_file = cfg.thompson_sampling.state_file
        self.last_daily_retrain_date: Dict[str, Optional[datetime.date]] = {sym: None for sym in cfg.symbols}
        self.bar_counters: Dict[str, int] = {sym: 0 for sym in cfg.symbols}
        self.load_state()

    def _calculate_rule_scale(self, symbol: str, context: Dict[str, Any]) -> float:
        """
        Computes a rule_scale (0,1] based on context variables.
        """
        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        # Extract context variables
        vol = context.get("vol", sym_state.last_atr) # Use last_atr if current vol not provided
        equity = context.get("equity", sym_state.current_equity)
        peak_equity = context.get("peak_equity", sym_state.peak_equity)
        max_drawdown = 1.0 - (equity / peak_equity) if peak_equity > 0 else 0.0
        consecutive_losses = sym_state.consecutive_losses

        rule_scale = 1.0

        # 1. Inverse Volatility Scale
        if vol > 0 and ts_cfg.vol_threshold > 0: # Avoid division by zero
            inverse_vol_scale = min(1.0, ts_cfg.vol_threshold / vol + 0.5) # Example scaling
            rule_scale *= inverse_vol_scale

        # 2. Drawdown Scale
        if max_drawdown > 0 and ts_cfg.dd_cut_multiplier > 0:
            drawdown_scale = max(0.1, 1.0 - ts_cfg.dd_cut_multiplier * max_drawdown)
            rule_scale *= drawdown_scale

        # 3. Consecutive Loss Scale
        if consecutive_losses > 0 and ts_cfg.consec_loss_cut > 0:
            consec_scale = max(0.1, 1.0 - ts_cfg.consec_loss_cut * consecutive_losses / 5.0) # Divide by 5 for example
            rule_scale *= consec_scale
        
        # Ensure rule_scale is within (0, 1]
        rule_scale = np.clip(rule_scale, 0.01, 1.0) # Min scale of 0.01 to avoid zeroing out

        logger.debug(f"[{symbol}] Rule Scale: {rule_scale:.2f} (Vol:{vol:.5f}, DD:{max_drawdown:.2%}, CL:{consecutive_losses})")
        return float(rule_scale)

    def get_params(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Samples discrete choices via Thompson Sampling and applies rule-based scaling.
        Returns a dict with chosen parameters and their discrete indices.
        """
        if not self.cfg.thompson_sampling.enabled:
            # If TS is disabled, return default risk parameters from cfg.risk
            return {
                "atr_multiplier_sl": self.cfg.risk.atr_multiplier_sl,
                "atr_multiplier_tp": self.cfg.risk.atr_multiplier_tp,
                "trailing_atr_mult": self.cfg.risk.trailing_atr_mult,
                "min_prob_long": self.cfg.risk.min_prob_long,
                "min_prob_short": self.cfg.risk.min_prob_short,
                "atr_idx": -1, # Indicate no TS choice
                "min_prob_idx": -1,
            }

        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        # Check and trigger reset if conditions are met
        self._check_and_trigger_reset(symbol, context, ensemble_auc=context.get("ensemble_auc", 0.5))

        # 1. Sample discrete choices (possibly contextual)
        atr_idx = None
        if getattr(self.cfg.thompson_sampling, "contextual_enabled", False) and sym_state.contextual_bandit is not None:
            # build a context vector from available context dict: vol, equity, peak_equity, ensemble_auc, adx, macd_diff, volatility_10, dist_from_ema_200
            # normalize vol by vol_threshold to keep scales reasonable
            vol = float(context.get("vol", sym_state.last_atr or 0.0))
            auc = float(context.get("ensemble_auc", 0.5))
            equity = float(context.get("equity", sym_state.current_equity or self.cfg.initial_equity))
            peak = float(context.get("peak_equity", sym_state.peak_equity or self.cfg.initial_equity))
            drawdown = 1.0 - (equity / peak) if peak > 0 else 0.0
            # time-of-day features (hour sin/cos)
            now = datetime.datetime.utcnow()
            hour = now.hour + now.minute / 60.0
            hour_sin = np.sin(2 * np.pi * hour / 24.0)
            hour_cos = np.cos(2 * np.pi * hour / 24.0)
            vol_scale = float(self.cfg.thompson_sampling.vol_threshold or 1e-6)

            # New context features
            adx = float(context.get("adx", 0.0))
            macd_diff = float(context.get("macd_diff", 0.0))
            volatility_10 = float(context.get("volatility_10", 0.0))
            dist_from_ema_200 = float(context.get("dist_from_ema_200", 0.0))

            current_x_features = [
                vol / max(vol_scale, 1e-9),
                auc,
                drawdown,
                hour_sin,
                hour_cos,
                adx / 100.0, # Normalize ADX (typically 0-100)
                macd_diff * 1000.0, # Scale macd_diff for better feature representation
                volatility_10 * 100.0, # Scale volatility
                dist_from_ema_200 * 100.0, # Scale distance
            ]

            # Add lagged vol if enabled
            if ts_cfg.enable_lagged_vol and len(sym_state.vol_history) > ts_cfg.lagged_vol_period:
                lagged_vol = sym_state.vol_history[-(ts_cfg.lagged_vol_period + 1)] # Get the lagged value
                current_x_features.append(lagged_vol / max(vol_scale, 1e-9))
            else:
                # Append a placeholder if not enabled or not enough history
                current_x_features.append(0.0)

            # Add vol-drawdown interaction if enabled
            if ts_cfg.enable_vol_drawdown_interaction:
                vol_drawdown_interaction = (vol / max(vol_scale, 1e-9)) * drawdown
                current_x_features.append(vol_drawdown_interaction)
            else:
                # Append a placeholder if not enabled
                current_x_features.append(0.0)

            x = np.array(current_x_features, dtype=float)

            # ensure dimension matches context_dim; if not, pad/truncate
            # Dynamically determine ctx_dim based on enabled features
            expected_ctx_dim = 9 # Base features
            if ts_cfg.enable_lagged_vol: expected_ctx_dim += 1
            if ts_cfg.enable_vol_drawdown_interaction: expected_ctx_dim += 1

            ctx_dim = int(getattr(self.cfg.thompson_sampling, "context_dim", expected_ctx_dim))

            if len(x) < ctx_dim:
                x = np.concatenate([x, np.zeros(ctx_dim - len(x))])
            elif len(x) > ctx_dim:
                x = x[:ctx_dim]
            atr_idx = sym_state.contextual_bandit.sample_arm(x)
        else:
            atr_idx = sym_state.atr_bandit.sample()

        min_prob_idx = sym_state.min_prob_bandit.sample()

        # 2. Apply rule-based scaling
        rule_scale = self._calculate_rule_scale(symbol, context)

        # Apply rule_scale to ATR-related parameters
        atr_choice = sym_state.atr_grid_values[atr_idx] * rule_scale
        # For min_prob, scaling might be different or not applied directly
        min_prob_choice = sym_state.min_prob_grid_values[min_prob_idx]

        # Note: For trailing_atr_mult, we can either optimize it with its own bandit
        # or scale it based on atr_choice or rule_scale. For simplicity, let's scale it with rule_scale
        trailing_atr_mult_choice = self.cfg.risk.trailing_atr_mult * rule_scale # Use default and scale

        # For min_prob_short, we can either optimize it with its own bandit
        # or assume it's the same as min_prob_long for simplicity
        min_prob_short_choice = min_prob_choice # Assuming symmetric for now

        logger.debug(f"[{symbol}] TS Params: ATR={atr_choice:.2f} (idx:{atr_idx}), MinProb={min_prob_choice:.2f} (idx:{min_prob_idx}), RuleScale={rule_scale:.2f}")

        # Exploration safety: if arm is under-visited, apply exploration risk multiplier
        ts_cfg = self.cfg.thompson_sampling
        is_exploratory = False
        exploration_risk_mult = 1.0
        if hasattr(sym_state.atr_bandit, "counts"):
            if sym_state.atr_bandit.counts[atr_idx] < ts_cfg.min_visits_for_exploration:
                is_exploratory = True
                exploration_risk_mult = float(ts_cfg.exploration_risk_mult)

        # Apply uncertainty-based risk scaling if enabled
        if ts_cfg.dynamic_uncertainty_risk_scaling_enabled and sym_state.contextual_bandit is not None:
            # Get the noise_var for the chosen arm
            chosen_arm_noise_var = sym_state.contextual_bandit.noise_var # This is the current dynamic noise_var
            if chosen_arm_noise_var > ts_cfg.uncertainty_threshold:
                uncertainty_penalty = (chosen_arm_noise_var / ts_cfg.uncertainty_threshold) * ts_cfg.uncertainty_risk_factor
                exploration_risk_mult *= max(0.0, 1.0 - uncertainty_penalty)
                logger.debug(f"[{symbol}] Uncertainty risk scaling applied: {exploration_risk_mult:.2f} (Noise Var: {chosen_arm_noise_var:.4f})")

        # Return chosen params plus exploratory metadata
        return {
            "atr_multiplier_sl": atr_choice,
            "atr_multiplier_tp": atr_choice,
            "trailing_atr_mult": trailing_atr_mult_choice,
            "min_prob_long": min_prob_choice,
            "min_prob_short": min_prob_short_choice,
            "atr_idx": atr_idx,
            "min_prob_idx": min_prob_idx,
            "rule_scale": rule_scale,
            "is_exploratory": is_exploratory,
            "exploration_risk_mult": exploration_risk_mult,
            "lagged_vol": current_x_features[9] if ts_cfg.enable_lagged_vol else 0.0, # Assuming index 9 for lagged_vol
            "vol_drawdown_interaction": current_x_features[10] if ts_cfg.enable_vol_drawdown_interaction else 0.0 # Assuming index 10 for interaction
        }

    def update_after_trade(self, symbol: str, trade: SimPosition):
        """
        Updates bandit statistics and rule state after a trade closes.
        """
        if not self.cfg.thompson_sampling.enabled:
            return

        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        # 1. Compute normalized reward (Sharpe-like: return per unit of risk)
        reward = 0.0
        if trade.pnl is not None and trade.risk_fraction is not None and trade.risk_fraction > 0 and trade.exit_equity is not None and trade.exit_equity > 0:
            # Calculate return per unit of risk
            raw_reward = trade.pnl / (trade.exit_equity * trade.risk_fraction)
            # Apply log utility and normalize
            shaped = np.sign(raw_reward) * np.log1p(abs(raw_reward))
            reward = float(np.clip(shaped / ts_cfg.reward_normalization_factor, -5.0, 5.0))
        elif trade.pnl is not None and trade.exit_equity is not None and trade.exit_equity > 0: # Fallback to simple PnL/Equity if risk_fraction is not available or zero
            raw_reward = trade.pnl / trade.entry_equity
            shaped = np.sign(raw_reward) * np.log1p(abs(raw_reward))
            reward = float(np.clip(shaped / ts_cfg.reward_normalization_factor, -5.0, 5.0))
        
        # Ensure reward is not NaN or Inf
        if not np.isfinite(reward):
            reward = 0.0

        # Apply drawdown-aware penalties to the reward
        current_drawdown = 1.0 - (sym_state.current_equity / sym_state.peak_equity) if sym_state.peak_equity > 0 else 0.0
        if current_drawdown >= ts_cfg.drawdown_penalty_threshold:
            penalty = current_drawdown * ts_cfg.drawdown_penalty_factor
            reward -= penalty
            logger.debug(f"[{symbol}] Drawdown penalty applied: -{penalty:.4f} (Current DD: {current_drawdown:.2%})")

        if sym_state.consecutive_losses >= ts_cfg.consecutive_loss_penalty_threshold:
            penalty = sym_state.consecutive_losses * ts_cfg.consecutive_loss_penalty_factor
            reward -= penalty
            logger.debug(f"[{symbol}] Consecutive loss penalty applied: -{penalty:.4f} (Consecutive Losses: {sym_state.consecutive_losses})")

        # Apply win rate reward / loss penalty
        if ts_cfg.enable_win_rate_reward and trade.pnl > 0:
            reward += ts_cfg.win_rate_reward_factor
            logger.debug(f"[{symbol}] Win rate bonus applied: +{ts_cfg.win_rate_reward_factor:.4f}")
        elif ts_cfg.enable_loss_penalty and trade.pnl <= 0:
            reward -= ts_cfg.loss_penalty_factor
            logger.debug(f"[{symbol}] Loss penalty applied: -{ts_cfg.loss_penalty_factor:.4f}")

        # 2. Update bandits
        if trade.atr_idx is not None and trade.atr_idx != -1:
            sym_state.atr_bandit.update(trade.atr_idx, reward, ts_cfg.decay)
            sym_state.atr_updates_since_last_adaptation += 1
        if trade.min_prob_idx is not None and trade.min_prob_idx != -1:
            sym_state.min_prob_bandit.update(trade.min_prob_idx, reward, ts_cfg.decay)
            sym_state.min_prob_updates_since_last_adaptation += 1

        # contextual update (if enabled)
        if getattr(ts_cfg, "contextual_enabled", False) and sym_state.contextual_bandit is not None:
            # reconstruct context from trade
            try:
                entry_time_raw = getattr(trade, "entry_time", None)
                if entry_time_raw is None:
                    entry_time = datetime.datetime.utcnow()
                elif isinstance(entry_time_raw, str):
                    entry_time = datetime.datetime.fromisoformat(entry_time_raw)
                else:
                    entry_time = entry_time_raw
                hour = entry_time.hour + entry_time.minute / 60.0
                hour_sin = np.sin(2 * np.pi * hour / 24.0)
                hour_cos = np.cos(2 * np.pi * hour / 24.0)
                vol = float(getattr(trade, "atr", sym_state.last_atr or 0.0))
                auc = float(getattr(trade, "entry_auc", 0.5))
                # Use entry_equity for drawdown calculation at entry time
                entry_equity = float(getattr(trade, "entry_equity", self.cfg.initial_equity))
                # Need to get peak_equity at entry time, which is not directly stored in trade. 
                # For simplicity, we'll use the current peak_equity, but ideally this would be historical.
                # This is a limitation for perfect historical context reconstruction in update_after_trade.
                drawdown_at_entry = 1.0 - (entry_equity / max(sym_state.peak_equity or self.cfg.initial_equity, 1e-9))
                vol_scale = float(ts_cfg.vol_threshold or 1e-6)

                # New context features from trade object
                adx = float(getattr(trade, "adx", 0.0))
                macd_diff = float(getattr(trade, "macd_diff", 0.0))
                volatility_10 = float(getattr(trade, "volatility_10", 0.0))
                dist_from_ema_200 = float(getattr(trade, "dist_from_ema_200", 0.0))

                current_x_features = [
                    vol / max(vol_scale, 1e-9),
                    auc,
                    drawdown_at_entry,
                    hour_sin,
                    hour_cos,
                    adx / 100.0, # Normalize ADX (typically 0-100)
                    macd_diff * 1000.0, # Scale macd_diff for better feature representation
                    volatility_10 * 100.0, # Scale volatility
                    dist_from_ema_200 * 100.0, # Scale distance
                ]

                # Add lagged vol if enabled (from trade object if available, otherwise from sym_state.vol_history)
                if ts_cfg.enable_lagged_vol:
                    # Ideally, lagged_vol should be stored in the trade object at entry.
                    # For now, we'll use a placeholder or assume it's not perfectly reconstructible for past trades.
                    # A more robust solution would be to store the full context vector in the trade object.
                    lagged_vol_from_trade = float(getattr(trade, "lagged_vol", 0.0))
                    current_x_features.append(lagged_vol_from_trade / max(vol_scale, 1e-9))
                else:
                    current_x_features.append(0.0)

                # Add vol-drawdown interaction if enabled
                if ts_cfg.enable_vol_drawdown_interaction:
                    vol_drawdown_interaction = (vol / max(vol_scale, 1e-9)) * drawdown_at_entry
                    current_x_features.append(vol_drawdown_interaction)
                else:
                    current_x_features.append(0.0)

                x = np.array(current_x_features, dtype=float)

                # Dynamically determine ctx_dim based on enabled features
                expected_ctx_dim = 9 # Base features
                if ts_cfg.enable_lagged_vol: expected_ctx_dim += 1
                if ts_cfg.enable_vol_drawdown_interaction: expected_ctx_dim += 1

                ctx_dim = int(getattr(ts_cfg, "context_dim", expected_ctx_dim))

                if len(x) < ctx_dim:
                    x = np.concatenate([x, np.zeros(ctx_dim - len(x))])
                elif len(x) > ctx_dim:
                    x = x[:ctx_dim]
                if trade.atr_idx is not None and trade.atr_idx != -1:
                    sym_state.contextual_bandit.update(int(trade.atr_idx), x, reward, ts_cfg.decay)
            except Exception:
                logger.exception("Contextual bandit update failed")

        # 3. Dynamic Grid Adaptation Check
        if ts_cfg.adaptive_grids_enabled:
            # Check ATR grid adaptation
            if sym_state.atr_updates_since_last_adaptation >= ts_cfg.adaptation_interval_updates:
                best_atr_arm_idx = int(np.argmax(sym_state.atr_bandit.sum_rewards / np.maximum(1.0, sym_state.atr_bandit.counts)))
                new_atr_grid = self._refine_grid(
                    sym_state.atr_grid_values, best_atr_arm_idx, ts_cfg.adaptation_refinement_factor,
                    ts_cfg.min_grid_size, ts_cfg.max_grid_size
                )
                if new_atr_grid != sym_state.atr_grid_values:
                    logger.info(f"[{symbol}] Adapting ATR grid. Old: {sym_state.atr_grid_values} -> New: {new_atr_grid}")
                    old_atr_bandit = sym_state.atr_bandit
                    old_atr_grid = sym_state.atr_grid_values
                    sym_state.atr_grid_values = new_atr_grid
                    sym_state.atr_bandit = ThompsonBandit(
                        num_arms=len(new_atr_grid), prior_mean=ts_cfg.prior_mean,
                        prior_var=ts_cfg.prior_var, min_var=1e-6
                    )
                    self._transfer_bandit_state(old_atr_bandit, old_atr_grid, sym_state.atr_bandit, new_atr_grid)
                    if sym_state.contextual_bandit is not None:
                        # Re-initialize contextual bandit with new num_arms
                        ctx_dim = int(getattr(ts_cfg, "context_dim", 5))
                        old_contextual_bandit = sym_state.contextual_bandit
                        sym_state.contextual_bandit = LinearThompson(num_arms=len(new_atr_grid), dim=ctx_dim, lambda_prior=1.0, noise_var=float(ts_cfg.obs_var or 1.0),
                                                                    dynamic_noise_var_enabled=ts_cfg.dynamic_noise_var_enabled, noise_var_window_size=ts_cfg.noise_var_window_size, min_noise_var=ts_cfg.min_noise_var,
                                                                    dynamic_uncertainty_risk_scaling_enabled=ts_cfg.dynamic_uncertainty_risk_scaling_enabled, uncertainty_risk_factor=ts_cfg.uncertainty_risk_factor, uncertainty_threshold=ts_cfg.uncertainty_threshold)
                        # Transfer state for contextual bandit (more complex, simple re-init for now)
                        # For LinearThompson, transferring A and b matrices would be ideal, but requires careful mapping.
                        # For simplicity, we'll re-initialize and let it learn on the new grid.
                        logger.warning(f"[{symbol}] Contextual bandit re-initialized due to ATR grid adaptation. Previous learning for contextual bandit is reset.")
                sym_state.atr_updates_since_last_adaptation = 0

            # Check MinProb grid adaptation
            if sym_state.min_prob_updates_since_last_adaptation >= ts_cfg.adaptation_interval_updates:
                best_min_prob_arm_idx = int(np.argmax(sym_state.min_prob_bandit.sum_rewards / np.maximum(1.0, sym_state.min_prob_bandit.counts)))
                new_min_prob_grid = self._refine_grid(
                    sym_state.min_prob_grid_values, best_min_prob_arm_idx, ts_cfg.adaptation_refinement_factor,
                    ts_cfg.min_grid_size, ts_cfg.max_grid_size
                )
                if new_min_prob_grid != sym_state.min_prob_grid_values:
                    logger.info(f"[{symbol}] Adapting MinProb grid. Old: {sym_state.min_prob_grid_values} -> New: {new_min_prob_grid}")
                    old_min_prob_bandit = sym_state.min_prob_bandit
                    old_min_prob_grid = sym_state.min_prob_grid_values
                    sym_state.min_prob_grid_values = new_min_prob_grid
                    sym_state.min_prob_bandit = ThompsonBandit(
                        num_arms=len(new_min_prob_grid), prior_mean=ts_cfg.prior_mean,
                        prior_var=ts_cfg.prior_var, min_var=1e-6
                    )
                    self._transfer_bandit_state(old_min_prob_bandit, old_min_prob_grid, sym_state.min_prob_bandit, new_min_prob_grid)
                sym_state.min_prob_updates_since_last_adaptation = 0

        # 4. Update rule state
        sym_state.current_equity = trade.exit_equity # Assuming SimPosition now tracks exit_equity
        sym_state.peak_equity = max(sym_state.peak_equity, sym_state.current_equity)
        sym_state.recent_returns.append(reward) # Append normalized reward
        sym_state.last_atr = trade.atr # Update last ATR for rule scaling context

        # Update vol_history for lagged features
        if ts_cfg.enable_lagged_vol:
            sym_state.vol_history.append(trade.atr) # Assuming trade.atr is the current vol

        if reward < 0:
            sym_state.consecutive_losses += 1
        else:
            sym_state.consecutive_losses = 0
        
        logger.debug(f"[{symbol}] Trade closed. Reward: {reward:.4f}. CL: {sym_state.consecutive_losses}")

    def save_state(self, open_positions_cache: dict | None = None):
        state_path = self.cfg.thompson_sampling.state_file
        try:
            symbol_states_data = {}
            for sym, sym_state in self.symbol_states.items():
                symbol_states_data[sym] = sym_state.get_state()

            state = {
                "symbol_states": symbol_states_data,
                "open_positions_cache": open_positions_cache or {},
                "last_daily_retrain_date": {sym: date.isoformat() if date else None for sym, date in self.last_daily_retrain_date.items()},
                "bar_counters": self.bar_counters
            }
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4, default=_json_serial)
            logger.info(f"RiskController state saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save RiskController state: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Failed to save RiskController state: {e}", level="ERROR")

    def load_state(self) -> dict:
        state_path = self.cfg.thompson_sampling.state_file
        if not os.path.exists(state_path):
            logger.info(f"No existing RiskController state file found at {state_path}. Starting fresh.")
            return {}

        try:
            with open(state_path, 'r') as f:
                state = json.load(f)

            symbol_states_data = state.get("symbol_states", {})
            for sym, sym_state_data in symbol_states_data.items():
                if sym in self.cfg.symbols: # Only load for active symbols
                    self.symbol_states[sym] = SymbolRiskState.from_state(self.cfg, sym_state_data)
                else:
                    logger.warning(f"State found for inactive symbol {sym}. Skipping load.")

            # Load last_daily_retrain_date and bar_counters
            loaded_dates = state.get("last_daily_retrain_date", {})
            for sym, date_str in loaded_dates.items():
                if sym in self.cfg.symbols:
                    self.last_daily_retrain_date[sym] = datetime.date.fromisoformat(date_str) if date_str else None
            self.bar_counters = state.get("bar_counters", {sym: 0 for sym in self.cfg.symbols})

            logger.info(f"RiskController state loaded from {state_path}")
            # Return the open positions cache if it exists
            return state.get("open_positions_cache", {})
        except Exception as e:
            logger.error(f"Failed to load RiskController state from {state_path}: {e}")
            if self.notifier: self.notifier.send_message(f"<b>ERROR:</b> Failed to load RiskController state from {state_path}: {e}", level="ERROR")
            return {}


    def diagnostics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of diagnostic information about the RiskController's state.
        """
        all_diagnostics = {}
        for sym, sym_state in self.symbol_states.items():
            sym_diag = {
                "peak_equity": sym_state.peak_equity,
                "current_equity": sym_state.current_equity,
                "consecutive_losses": sym_state.consecutive_losses,
                "recent_returns_count": len(sym_state.recent_returns),
                "atr_bandit_counts": sym_state.atr_bandit.counts.tolist(),
                "atr_bandit_sum_rewards": sym_state.atr_bandit.sum_rewards.tolist(),
                "min_prob_bandit_counts": sym_state.min_prob_bandit.counts.tolist(),
                "min_prob_bandit_sum_rewards": sym_state.min_prob_bandit.sum_rewards.tolist(),
                "atr_grid_values": sym_state.atr_grid_values,
                "min_prob_grid_values": sym_state.min_prob_grid_values,
            }
            all_diagnostics[sym] = sym_diag
        return all_diagnostics

    def _reset_bandit_state(self, symbol: str, current_time: datetime.datetime):
        sym_state = self.symbol_states[symbol]
        ts_cfg = self.cfg.thompson_sampling

        logger.warning(f"[{symbol}] Triggering bandit reset due to performance degradation or market shift.")
        if self.notifier: self.notifier.send_message(f"<b>RISK ALERT:</b> [{symbol}] Bandit reset triggered!", level="WARNING")

        # Reset ThompsonBandits to initial state
        sym_state.atr_bandit = ThompsonBandit(
            num_arms=len(ts_cfg.atr_grid),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )
        sym_state.min_prob_bandit = ThompsonBandit(
            num_arms=len(ts_cfg.min_prob_grid),
            prior_mean=ts_cfg.prior_mean,
            prior_var=ts_cfg.prior_var,
            min_var=1e-6
        )

        # Reset contextual bandit if enabled
        if getattr(ts_cfg, "contextual_enabled", False) and sym_state.contextual_bandit is not None:
            ctx_dim = int(getattr(ts_cfg, "context_dim", 9))
            sym_state.contextual_bandit = LinearThompson(
                num_arms=len(ts_cfg.atr_grid),
                dim=ctx_dim,
                lambda_prior=1.0,
                noise_var=float(ts_cfg.obs_var or 1.0),
                dynamic_noise_var_enabled=ts_cfg.dynamic_noise_var_enabled,
                noise_var_window_size=ts_cfg.noise_var_window_size,
                min_noise_var=ts_cfg.min_noise_var,
                dynamic_uncertainty_risk_scaling_enabled=ts_cfg.dynamic_uncertainty_risk_scaling_enabled,
                uncertainty_risk_factor=ts_cfg.uncertainty_risk_factor,
                uncertainty_threshold=ts_cfg.uncertainty_threshold
            )

        # Reset dynamic grids to initial config values
        sym_state.atr_grid_values = list(ts_cfg.atr_grid)
        sym_state.min_prob_grid_values = list(ts_cfg.min_prob_grid)

        # Reset adaptation counters
        sym_state.atr_updates_since_last_adaptation = 0
        sym_state.min_prob_updates_since_last_adaptation = 0

        # Reset performance metrics for the symbol
        sym_state.peak_equity = sym_state.current_equity # Reset peak to current equity
        sym_state.consecutive_losses = 0
        sym_state.recent_returns.clear()

        # Set last reset time for cooldown
        sym_state.last_reset_time = current_time
        logger.info(f"[{symbol}] Bandit reset complete. Cooldown until {current_time + datetime.timedelta(hours=ts_cfg.reset_cooldown_hours)}")

    def _check_and_trigger_reset(self, symbol: str, context: Dict[str, Any], ensemble_auc: float):
        ts_cfg = self.cfg.thompson_sampling
        if not ts_cfg.bandit_reset_enabled:
            return

        sym_state = self.symbol_states[symbol]
        current_time = datetime.datetime.utcnow()

        # Check cooldown
        if sym_state.last_reset_time:
            cooldown_end_time = sym_state.last_reset_time + datetime.timedelta(hours=ts_cfg.reset_cooldown_hours)
            if current_time < cooldown_end_time:
                logger.debug(f"[{symbol}] Bandit reset in cooldown until {cooldown_end_time}")
                return

        # Check triggers
        reset_triggered = False
        trigger_reason = ""

        # 1. Drawdown trigger
        equity = context.get("equity", sym_state.current_equity)
        peak_equity = context.get("peak_equity", sym_state.peak_equity)
        if peak_equity > 0:
            current_drawdown = 1.0 - (equity / peak_equity)
            if current_drawdown >= ts_cfg.reset_on_drawdown_percent:
                reset_triggered = True
                trigger_reason = f"Drawdown ({current_drawdown:.2%}) exceeded {ts_cfg.reset_on_drawdown_percent:.2%}"

        # 2. Consecutive losses trigger
        if not reset_triggered and sym_state.consecutive_losses >= ts_cfg.reset_on_consecutive_losses:
            reset_triggered = True
            trigger_reason = f"Consecutive losses ({sym_state.consecutive_losses}) exceeded {ts_cfg.reset_on_consecutive_losses}"

        # 3. Low ensemble AUC trigger
        if not reset_triggered and ensemble_auc < ts_cfg.reset_on_low_ensemble_auc:
            reset_triggered = True
            trigger_reason = f"Ensemble AUC ({ensemble_auc:.4f}) below {ts_cfg.reset_on_low_ensemble_auc:.4f}"
        
        if reset_triggered:
            logger.warning(f"[{symbol}] Bandit reset triggered: {trigger_reason}")
            self._reset_bandit_state(symbol, current_time)

    @staticmethod
    def _transfer_bandit_state(old_bandit: ThompsonBandit, old_grid: List[float], new_bandit: ThompsonBandit, new_grid: List[float]):
        """
        Transfers learned statistics from an old bandit to a new bandit with a refined grid.
        Maps old arm values to the closest new arm values.
        """
        if not old_grid or not new_grid:
            return

        # For each old arm, find the closest new arm and transfer its statistics
        for old_idx, old_val in enumerate(old_grid):
            if old_bandit.counts[old_idx] > 0:
                # Find the index of the closest value in the new grid
                new_idx = int(np.argmin(np.abs(np.array(new_grid) - old_val)))

                # Transfer statistics. If multiple old arms map to the same new arm,
                # their statistics will be summed up. This is a reasonable heuristic.
                new_bandit.counts[new_idx] += old_bandit.counts[old_idx]
                new_bandit.sum_rewards[new_idx] += old_bandit.sum_rewards[old_idx]
                new_bandit.sum_squared_rewards[new_idx] += old_bandit.sum_squared_rewards[old_idx]

        logger.debug(f"Transferred bandit state from {len(old_grid)} to {len(new_grid)} arms.")

    @staticmethod
    def _refine_grid(current_grid: List[float], best_arm_index: int, refinement_factor: float, min_grid_size: int, max_grid_size: int) -> List[float]:
        if not (0 < refinement_factor < 1):
            logger.warning(f"Invalid refinement_factor: {refinement_factor}. Must be between 0 and 1. Using 0.5.")
            refinement_factor = 0.5

        if not current_grid or len(current_grid) < min_grid_size:
            return current_grid # Cannot refine or already too small

        best_val = current_grid[best_arm_index]

        # Determine the interval around the best value
        # If best_arm_index is 0, use the interval to the next arm
        # If best_arm_index is last, use the interval to the previous arm
        # Otherwise, use the smaller of the two adjacent intervals
        lower_bound = current_grid[best_arm_index - 1] if best_arm_index > 0 else best_val - (current_grid[1] - current_grid[0])
        upper_bound = current_grid[best_arm_index + 1] if best_arm_index < len(current_grid) - 1 else best_val + (current_grid[-1] - current_grid[-2])

        # Ensure bounds are sensible if at edges
        if best_arm_index == 0:
            lower_bound = best_val - (current_grid[1] - best_val) * 2 # Extend a bit below
        if best_arm_index == len(current_grid) - 1:
            upper_bound = best_val + (best_val - current_grid[-2]) * 2 # Extend a bit above

        # Calculate the new, narrower range
        current_range = upper_bound - lower_bound
        new_range = current_range * refinement_factor
        
        # Center the new range around the best_val
        new_lower = best_val - new_range / 2
        new_upper = best_val + new_range / 2

        # Ensure new_lower and new_upper don't go beyond the original min/max of the full grid
        original_min = min(current_grid)
        original_max = max(current_grid)
        new_lower = max(new_lower, original_min)
        new_upper = min(new_upper, original_max)

        # If the new range is too small or invalid, return current grid
        if new_upper <= new_lower:
            logger.debug(f"Refinement resulted in invalid range [{new_lower}, {new_upper}]. Returning current grid.")
            return current_grid

        # Generate new grid points
        num_new_points = min(max_grid_size, max(min_grid_size, len(current_grid) + 2)) # Add a few points, but respect max_grid_size
        new_grid = np.linspace(new_lower, new_upper, num_new_points).tolist()
        new_grid = sorted(list(set(new_grid + [best_val]))) # Ensure best_val is always in the new grid
        
        # Ensure grid size constraints are met
        if len(new_grid) < min_grid_size:
            # If after refinement, grid is too small, try to expand it slightly or just return original
            logger.debug(f"Refined grid size {len(new_grid)} is less than min_grid_size {min_grid_size}. Returning current grid.")
            return current_grid
        if len(new_grid) > max_grid_size:
            # If too large, resample to max_grid_size
            new_grid = np.linspace(min(new_grid), max(new_grid), max_grid_size).tolist()

        logger.info(f"Grid refined from {len(current_grid)} to {len(new_grid)} arms. Old best: {best_val:.4f}. New range: [{min(new_grid):.4f}, {max(new_grid):.4f}]")
        return new_grid
