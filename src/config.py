# src/config.py
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class FeatureCfg:
    rsi_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    window_vol: int = 20
    roc_lags: List[int] = field(default_factory=lambda: [1,3,5,10])

@dataclass
class RiskCfg:
    risk_per_trade: float = 0.005
    max_positions: int = 3
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 2.5
    trailing_atr_mult: float = 1.0
    min_prob_long: float = 0.55
    min_prob_short: float = 0.55
    block_on_drawdown: float = 0.10
    session_filter: Dict[str, str] | None = None

@dataclass
class Cfg:
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "M5"
    history_bars: int = 2000
    retrain_every_bars: int = 250
    prediction_horizon: int = 6
    features: FeatureCfg = field(default_factory=FeatureCfg)
    models: List[Dict[str, Any]] = field(default_factory=list)
    ensemble: Dict[str, Any] = field(default_factory=dict)
    risk: RiskCfg = field(default_factory=RiskCfg)
    logging: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "Cfg":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return Cfg(
            symbols=raw.get("symbols", ["EURUSD"]),
            timeframe=raw.get("timeframe", "M5"),
            history_bars=raw.get("history_bars", 2000),
            retrain_every_bars=raw.get("retrain_every_bars", 250),
            prediction_horizon=raw.get("prediction_horizon", 6),
            features=FeatureCfg(**raw.get("features", {})),
            models=raw.get("models", []),
            ensemble=raw.get("ensemble", {}),
            risk=RiskCfg(**raw.get("risk", {})),
            logging=raw.get("logging", {}),
        )
