# src/config.py
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureCfg:
    rsi_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    window_vol: int = 20
    roc_lags: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    adx_period: int = 14
    rsi_ob_level: int = 70
    rsi_os_level: int = 30
    adx_trend_thresh: int = 25
    timeframe_minutes: int = 5

@dataclass
class RiskCfg:
    # default static risk values (can be overridden by YAML)
    risk_per_trade: float = 0.005
    max_positions: int = 3
    max_portfolio_risk: float = 0.03
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 2.5
    breakeven_at_1R: bool = True
    trailing_atr_mult: float = 1.0
    min_prob_long: float = 0.55
    min_prob_short: float = 0.55
    block_on_drawdown: float = 0.10
    transaction_cost_pips: float = 1.5
    session_filter: Optional[Dict[str, str]] = None
    min_ensemble_auc: float = 0.55
    min_auc_improvement: float = 0.005
    max_drawdown_for_pruning: float = 0.70 # New: Max drawdown allowed before Optuna trial pruning
    dynamic_risk: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "base_risk": 0.005,
            "max_risk": 0.01,
            "auc_floor": 0.55,
            "auc_ceiling": 0.65,
        }
    )
    dynamic_tp: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "base_tp_mult": 2.0,
            "max_tp_mult": 3.5,
            "auc_floor": 0.55,
            "auc_ceiling": 0.65,
        }
    )

@dataclass
class WatchdogCfg:
    max_consecutive_losses: int = 5
    cooldown_hours: float = 1.0
    # additional optional thresholds
    daily_loss_limit: Optional[float] = None  # absolute or fraction of equity (if used)

@dataclass
class Cfg:
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    timeframe: str = "M5"
    history_bars: int = 2000
    retrain_every_bars: int = 250
    prediction_horizon: int = 6
    data_source: str = "csv"
    use_gpu: bool = False
    cv_samples_per_split: int = 300
    optuna_n_trials: int = 150
    optuna_pruning_interval: int = 100 # New: Interval for Optuna pruning checks
    n_jobs: int = -1 # Number of parallel jobs for tuning. -1 means all available CPU cores.
    features: FeatureCfg = field(default_factory=FeatureCfg)
    models: List[Dict[str, Any]] = field(default_factory=list)
    ensemble: Dict[str, Any] = field(default_factory=dict)
    risk: RiskCfg = field(default_factory=RiskCfg)
    logging: Dict[str, Any] = field(default_factory=dict)
    watchdog: WatchdogCfg = field(default_factory=WatchdogCfg)

    def timeframe_seconds(self) -> Optional[int]:
        """ Convert timeframe string like 'M5', 'H1', 'D1' to seconds.
        Returns None for unknown formats.
        """
        if not self.timeframe:
            return None
        tf = str(self.timeframe).upper().strip()
        try:
            unit = tf[0]
            value = int(tf[1:])
            if unit == "M":
                return int(value * 60)
            if unit == "H":
                return int(value * 3600)
            if unit == "D":
                return int(value * 86400)
        except Exception:
            logger.warning(f"Cfg: invalid timeframe format '{self.timeframe}'")
        return None

    def timeframe_minutes(self) -> Optional[int]:
        """ Convert timeframe string like 'M5', 'H1', 'D1' to minutes.
        Returns None for unknown formats.
        """
        seconds = self.timeframe_seconds()
        if seconds is not None:
            return seconds // 60
        return None

    @staticmethod
    def from_yaml(path: str) -> "Cfg":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # features may contain lists (for tuning); pick sensible defaults
        raw_features = raw.get("features", {}) or {}
        cleaned_features: Dict[str, Any] = {}
        for k, v in raw_features.items():
            if isinstance(v, list) and k != "roc_lags":
                cleaned_features[k] = v[0]
            else:
                cleaned_features[k] = v

        try:
            features_obj = FeatureCfg(**cleaned_features)
        except Exception as e:
            logger.warning(f"Invalid feature config in YAML: {e}; using defaults.")
            features_obj = FeatureCfg()

        try:
            risk_obj = RiskCfg(**(raw.get("risk", {}) or {}))
        except Exception as e:
            logger.warning(f"Invalid risk config in YAML: {e}; using defaults.")
            risk_obj = RiskCfg()

        # parse watchdog block if present
        try:
            wd_raw = raw.get("watchdog", {}) or {}
            watchdog_obj = WatchdogCfg(**wd_raw) if wd_raw else WatchdogCfg()
        except Exception as e:
            logger.warning(f"Invalid watchdog config in YAML: {e}; using defaults.")
            watchdog_obj = WatchdogCfg()

        return Cfg(
            symbols=raw.get("symbols", ["EURUSD"]),
            timeframe=raw.get("timeframe", "M5"),
            history_bars=int(raw.get("history_bars", 2000)),
            retrain_every_bars=int(raw.get("retrain_every_bars", 250)),
            prediction_horizon=int(raw.get("prediction_horizon", 6)),
            data_source=raw.get("data_source", "csv"),
            use_gpu=bool(raw.get("use_gpu", False)),
            cv_samples_per_split=int(raw.get("cv_samples_per_split", 300)),
            optuna_n_trials=int(raw.get("optuna_n_trials", 100)),
            optuna_pruning_interval=int(raw.get("optuna_pruning_interval", 100)), # New
            n_jobs=int(raw.get("n_jobs", -1)), # New
            features=features_obj,
            models=raw.get("models", []),
            ensemble=raw.get("ensemble", {}),
            risk=risk_obj,
            logging=raw.get("logging", {}),
            watchdog=watchdog_obj,
        )
