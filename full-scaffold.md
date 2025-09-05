# XM–MT5 AI/ML Trading Bot (Python)

**Disclaimer:** For education/testing on demo only. Trading involves risk. This project connects to XM-MT5 demo via the `MetaTrader5` Python package, builds rich features, trains multiple ML models, ensembles their probabilities, backtests with walk-forward validation, and executes/risk-manages live demo trades.

---

## Project Structure

```
mt5-ml-bot/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ main.py
├─ config.yaml
├─ data/
│  └─ cache/
├─ models/
├─ logs/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ mt5_client.py
│  ├─ data.py
│  ├─ features.py        # (your provided file + small robustness tweaks)
│  ├─ labels.py
│  ├─ strategy_base.py
│  ├─ strategy_ml.py     # (your provided file, extended)
│  ├─ ensemble.py        # soft-vote + stacking + risk-adjusted weighting
│  ├─ risk.py            # position sizing, SL/TP, trailing stops
│  ├─ execution.py       # MT5 order routing helpers
│  ├─ backtest.py        # vectorized walk-forward backtest
│  ├─ trainer.py         # train, tune, save, and load models
│  └─ utils.py           # common helpers (logging, timezones, etc.)
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then edit credentials
```

**requirements.txt**

```txt
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
lightgbm>=4.3
xgboost>=2.1
ta>=0.11
scipy>=1.13
joblib>=1.4
pyyaml>=6.0
python-dotenv>=1.0
MetaTrader5>=5.0
loguru>=0.7
optuna>=3.6
```

**.env.example**

```env
# XM-MT5 demo login (fill these in for demo account)
MT5_LOGIN=12345678
MT5_PASSWORD=YOUR_DEMO_PASSWORD
MT5_SERVER=XMGlobal-Demo 5
MT5_PATH=  # optional, path to terminal64.exe if auto-detection fails
TZ=Etc/UTC  # or Asia/Manila
```

**config.yaml** (edit pairs/timeframes/risk)

```yaml
symbols: ["EURUSD", "XAUUSD"]
timeframe: "M5"           # One of: M1, M5, M15, M30, H1, H4
history_bars: 2000         # how many bars to pull when (re)starting
retrain_every_bars: 250    # walk-forward window
prediction_horizon: 6      # bars ahead for label (e.g., ~30min on M5)
# Feature params
features:
  rsi_period: 14
  ema_fast: 12
  ema_slow: 26
  window_vol: 20
  roc_lags: [1, 3, 5, 10]
# Models (base learners)
models:
  - name: lgbm
    params: {n_estimators: 400, learning_rate: 0.03, subsample: 0.8, colsample_bytree: 0.8}
  - name: xgb
    params: {n_estimators: 400, learning_rate: 0.05, subsample: 0.8, colsample_bytree: 0.8, max_depth: 5}
  - name: rf
    params: {n_estimators: 400, max_depth: 8, min_samples_leaf: 3}
  - name: logreg
    params: {}
# Ensemble
ensemble:
  method: stacking   # one of: soft_vote, stacking, risk_weighted
  weights: {lgbm: 0.35, xgb: 0.35, rf: 0.2, logreg: 0.1}
  meta: {type: "logit", C: 1.0}
# Risk
risk:
  risk_per_trade: 0.005      # 0.5% of equity
  max_positions: 3
  atr_multiplier_sl: 1.5
  atr_multiplier_tp: 2.5
  trailing_atr_mult: 1.0
  min_prob_long: 0.55        # ensemble p_up threshold for long
  min_prob_short: 0.55       # 1-p_up threshold for short
  block_on_drawdown: 0.1     # pause if 10% DD from peak
  session_filter:            # optional: only trade these hours exchange-time
    start: "07:00"
    end: "22:00"
logging:
  level: INFO
  to_file: true
  rotate: 10 MB
  retention: 7 days
```

---

## Core Code

### `src/config.py`

```python
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
    symbols: List[str] = None
    timeframe: str = "M5"
    history_bars: int = 2000
    retrain_every_bars: int = 250
    prediction_horizon: int = 6
    features: FeatureCfg = FeatureCfg()
    models: List[Dict[str, Any]] = None
    ensemble: Dict[str, Any] = None
    risk: RiskCfg = RiskCfg()
    logging: Dict[str, Any] = None

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
```

### `src/mt5_client.py`

```python
from __future__ import annotations
import os
from typing import Optional
import MetaTrader5 as mt5
from loguru import logger

class MT5Client:
    def __init__(self, login: int, password: str, server: str, path: Optional[str] = None):
        self.login = int(login)
        self.password = password
        self.server = server
        self.path = path

    def connect(self) -> bool:
        if not mt5.initialize(path=self.path or None):
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
        else:
            logger.info("MT5 login OK")
        return authorized

    def shutdown(self):
        mt5.shutdown()
```

### `src/data.py`

```python
from __future__ import annotations
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from loguru import logger

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

def fetch_bars(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    tf = TF_MAP[timeframe]
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol} {timeframe}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})


def merge_features_labels(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    out = X.copy()
    out["y"] = y
    out["close"] = df["close"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["volume"] = df.get("volume")
    return out.dropna()
```

### `src/features.py` (your code + tiny tweaks)

```python
import pandas as pd
import numpy as np
import ta

DEF_FILL = {"method": "ffill"}

class FeatureConfig:
    def __init__(self, rsi_period=14, ema_fast=12, ema_slow=26, window_vol=20, roc_lags=(1,3,5,10)):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.window_vol = window_vol
        self.roc_lags = list(roc_lags)


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    X["rsi"] = ta.momentum.rsi(df["close"], window=cfg.rsi_period)

    X["ema_fast"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_fast)
    X["ema_slow"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_slow)
    X["ema_diff"] = (X["ema_fast"] - X["ema_slow"]) / df["close"]

    bb = ta.volatility.BollingerBands(df["close"], window=cfg.window_vol, window_dev=2)
    X["bb_high"] = bb.bollinger_hband()
    X["bb_low"] = bb.bollinger_lband()
    X["bb_width"] = (X["bb_high"] - X["bb_low"]) / df["close"]

    macd = ta.trend.MACD(df["close"])
    X["macd"] = macd.macd()
    X["macd_signal"] = macd.macd_signal()
    X["macd_diff"] = macd.macd_diff()

    for l in cfg.roc_lags:
        X[f"ret_{l}"] = df["close"].pct_change(l)

    X["volatility_10"] = df["close"].pct_change().rolling(10).std()
    X["volatility_20"] = df["close"].pct_change().rolling(20).std()

    if "volume" in df.columns:
        X["vol_ma_20"] = df["volume"].rolling(20).mean()
        X["vol_ratio"] = df["volume"] / (X["vol_ma_20"] + 1e-10)

    X["momentum_5"] = df["close"] - df["close"].shift(5)
    X["momentum_10"] = df["close"] - df["close"].shift(10)

    X["atr_14"] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    X["fractal_up"] = ((df["high"].shift(2) < df["high"].shift(1)) &
                        (df["high"].shift(1) > df["high"]) &
                        (df["high"].shift(1) > df["high"].shift(-1)) &
                        (df["high"].shift(1) > df["high"].shift(-2))).astype(int)
    X["fractal_down"] = ((df["low"].shift(2) > df["low"].shift(1)) &
                          (df["low"].shift(1) < df["low"]) &
                          (df["low"].shift(1) < df["low"].shift(-1)) &
                          (df["low"].shift(1) < df["low"].shift(-2))).astype(int)

    X["sentiment_score"] = 0.0

    X["ret_skew_10"] = df["close"].pct_change().rolling(10).skew()
    X["ret_kurt_10"] = df["close"].pct_change().rolling(10).kurt()

    X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    return X


def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = (fwd > 0).astype(int)
    return y.loc[df.index]
```

### `src/labels.py`

```python
import pandas as pd

def binary_up_down(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int)
```

### `src/strategy_base.py`

```python
from __future__ import annotations
import pandas as pd

class Strategy:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def online_update(self, X_new: pd.DataFrame, y_new: pd.Series, X_hist=None, y_hist=None):
        pass
```

### `src/strategy_ml.py` (extended to keep your API and add calibration)

```python
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from .strategy_base import Strategy

class MLStrategy(Strategy):
    def __init__(self, model="lgbm", random_state=42, calibrate=True, **kwargs):
        self.model_name = model
        self.random_state = random_state
        self.calibrate = calibrate

        if model == "rf":
            base = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                min_samples_leaf=kwargs.get("min_samples_leaf", 3),
                n_jobs=-1,
                random_state=random_state,
            )
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])
        elif model == "xgb":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed")
            base = XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            )
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])
        elif model == "lgbm":
            if LGBMClassifier is None:
                raise ImportError("lightgbm is not installed")
            base = LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", -1),
                learning_rate=kwargs.get("learning_rate", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                min_child_samples=kwargs.get("min_child_samples", 5),
                random_state=random_state,
                n_jobs=-1,
            )
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])
        elif model == "logreg":
            # Online-friendly linear model
            sgd = SGDClassifier(loss="log_loss", learning_rate="optimal", max_iter=1, tol=None, warm_start=True,
                                random_state=random_state)
            self.supports_online = True
            self._pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", sgd)])
        else:
            raise ValueError(f"Unknown model '{model}'")

        self._calibrator = None

    def _sanitize(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xc = self._sanitize(X)
        yc = y.loc[Xc.index]
        if len(Xc) == 0 or len(yc) == 0:
            raise ValueError("Empty dataset after sanitization. Cannot fit ML model.")

        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(Xc)//300)))
        scores = []
        for tr, va in tscv.split(Xc):
            self._pipe.fit(Xc.iloc[tr], yc.iloc[tr])
            p = self._proba_raw(Xc.iloc[va])
            scores.append(roc_auc_score(yc.iloc[va], p))

        self._pipe.fit(Xc, yc)
        self.cv_auc_ = float(np.mean(scores)) if scores else None

        if self.calibrate:
            self._calibrator = CalibratedClassifierCV(self._pipe.named_steps["clf"], cv="prefit", method="isotonic")
            self._calibrator.fit(Xc, yc)
        return self

    def _proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self._pipe, "predict_proba") and hasattr(self._pipe.named_steps["clf"], "predict_proba"):
            return self._pipe.predict_proba(X)[:, 1]
        if hasattr(self._pipe.named_steps["clf"], "decision_function"):
            from sklearn.metrics import log_loss
            # Convert margin to probability via logistic link (approx)
            dec = self._pipe.named_steps["clf"].decision_function(X)
            return 1 / (1 + np.exp(-dec))
        return self._pipe.predict_proba(X)[:, 1]

    def online_update(self, X_new: pd.DataFrame, y_new: pd.Series, X_hist: pd.DataFrame = None, y_hist: pd.Series = None):
        Xn = self._sanitize(X_new)
        yn = y_new.loc[Xn.index]
        if len(Xn) == 0:
            return
        if self.supports_online:
            self._pipe.named_steps["clf"].partial_fit(Xn, yn, classes=[0,1])
            if self.calibrate:
                # refresh calibration lightly using recent data
                self._calibrator = CalibratedClassifierCV(self._pipe.named_steps["clf"], cv=3, method="isotonic")
                self._calibrator.fit(Xn, yn)
        else:
            if X_hist is not None and y_hist is not None:
                Xc = pd.concat([X_hist, Xn]).loc[~pd.concat([X_hist, Xn]).index.duplicated(keep='last')]
                yc = pd.concat([y_hist, yn]).loc[Xc.index]
            else:
                Xc, yc = Xn, yn
            self.fit(Xc, yc)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        Xc = self._sanitize(X)
        if len(Xc) == 0:
            return pd.Series(0.5, index=X.index, name="p_up")
        if self.calibrate and self._calibrator is not None:
            p = self._calibrator.predict_proba(Xc)[:,1]
        else:
            p = self._proba_raw(Xc)
        return pd.Series(p, index=Xc.index, name="p_up")
```

### `src/ensemble.py`

```python
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List
from .strategy_ml import MLStrategy

class Ensemble:
    def __init__(self, cfg):
        self.cfg = cfg
        self.members: Dict[str, MLStrategy] = {}
        for m in cfg.models:
            name = m["name"]
            params = m.get("params", {})
            self.members[name] = MLStrategy(model=name, **params)
        self.method = cfg.ensemble.get("method", "soft_vote")
        self.weights = cfg.ensemble.get("weights", {k:1/len(self.members) for k in self.members})
        self.meta = cfg.ensemble.get("meta", {"type":"logit","C":1.0})
        self._stacker = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Ps = []
        for name, model in self.members.items():
            model.fit(X, y)
            Ps.append(model.predict_proba(X).rename(name))
        P = pd.concat(Ps, axis=1)
        if self.method == "stacking":
            from sklearn.linear_model import LogisticRegression
            self._stacker = LogisticRegression(C=self.meta.get("C",1.0), max_iter=200)
            self._stacker.fit(P.values, y.loc[P.index].values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        Pcols = []
        for name, model in self.members.items():
            Pcols.append(model.predict_proba(X).rename(name))
        P = pd.concat(Pcols, axis=1)
        if self.method == "soft_vote":
            w = np.array([self.weights.get(k, 1.0) for k in P.columns], dtype=float)
            w = w / w.sum()
            p = (P.values * w).sum(axis=1)
        elif self.method == "stacking" and self._stacker is not None:
            p = self._stacker.predict_proba(P.values)[:,1]
        elif self.method == "risk_weighted":
            # Weight by each model's recent Sharpe-like score (toy implementation)
            eps = 1e-9
            mus = P.rolling(200).mean()
            sig = P.rolling(200).std()
            score = (mus / (sig + eps)).iloc[-1].fillna(0.0)
            w = (score.clip(lower=0) + eps).values
            if w.sum() == 0:
                w = np.ones_like(w)
            w = w / w.sum()
            p = (P.values * w).sum(axis=1)
        else:
            p = P.mean(axis=1).values
        return pd.Series(p, index=P.index, name="p_up")
```

### `src/risk.py`

```python
from __future__ import annotations
import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.equity_peak = None

    def position_size(self, equity: float, atr: float, pip_value: float, pip_size: float) -> float:
        risk_amt = equity * self.cfg.risk_per_trade
        # Position sizing based on ATR distance to SL
        sl_distance = self.cfg.atr_multiplier_sl * atr
        if sl_distance <= 0:
            return 0.0
        units = risk_amt / (sl_distance * pip_value)
        # convert to lots (forex standard lot = 100k units for FX); for gold use contract size
        lots = max(0.01, min(units * pip_size, 5.0))  # clamp between 0.01 and 5 lots
        return round(lots, 2)

    def stop_targets(self, price: float, atr: float, direction: str):
        sl_mult = self.cfg.atr_multiplier_sl
        tp_mult = self.cfg.atr_multiplier_tp
        if direction == "long":
            sl = price - sl_mult * atr
            tp = price + tp_mult * atr
        else:
            sl = price + sl_mult * atr
            tp = price - tp_mult * atr
        return sl, tp

    def should_trade(self, now_local: pd.Timestamp, dd: float) -> bool:
        if dd >= self.cfg.block_on_drawdown:
            return False
        sess = self.cfg.session_filter
        if not sess:
            return True
        start = pd.to_datetime(sess["start"]).time()
        end = pd.to_datetime(sess["end"]).time()
        return start <= now_local.time() <= end
```

### `src/execution.py`

```python
from __future__ import annotations
from dataclasses import dataclass
import MetaTrader5 as mt5
from loguru import logger

@dataclass
class OrderResult:
    ok: bool
    ticket: int | None
    message: str

class Execution:
    def market_order(self, symbol: str, lots: float, direction: str, sl: float | None, tp: float | None) -> OrderResult:
        type_map = {"long": mt5.ORDER_TYPE_BUY, "short": mt5.ORDER_TYPE_SELL}
        order_type = type_map[direction]
        price = mt5.symbol_info_tick(symbol).ask if direction == "long" else mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 424242,
            "comment": "ml-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(False, None, str(mt5.last_error()))
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} {result.comment}")
            return OrderResult(False, None, f"{result.retcode} {result.comment}")
        logger.info(f"Order OK ticket={result.order}")
        return OrderResult(True, result.order, "OK")
```

### `src/backtest.py`

```python
from __future__ import annotations
import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, ensemble, risk_mgr, horizon: int):
        self.ens = ensemble
        self.risk = risk_mgr
        self.h = horizon

    def walk_forward(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, step: int = 250, start: int = 500):
        equity = 10000.0
        peak = equity
        curve = []
        i = start
        while i < len(X) - self.h:
            Xtr, ytr = X.iloc[:i], y.iloc[:i]
            Xte = X.iloc[i:i+step]
            self.ens.fit(Xtr, ytr)
            p = self.ens.predict_proba(Xte)
            # simple strategy: long if p>=0.55, short if p<=0.45, else flat
            ret = df["close"].pct_change(self.h).shift(-self.h).loc[p.index]
            sig = np.where(p >= 0.55, 1, np.where(p <= 0.45, -1, 0))
            pnl = sig * ret
            equity *= (1 + np.nanmean(pnl))
            peak = max(peak, equity)
            curve.append((Xte.index[-1], equity, (peak - equity)/peak))
            i += step
        ec = pd.DataFrame(curve, columns=["time","equity","dd"]).set_index("time")
        return ec
```

### `src/trainer.py`

```python
from __future__ import annotations
import joblib
from pathlib import Path
from .ensemble import Ensemble

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit_save(self, X, y, outdir: str = "models"):
        ens = Ensemble(self.cfg)
        ens.fit(X, y)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        joblib.dump(ens, f"{outdir}/ensemble.pkl")
        return ens

    @staticmethod
    def load(path: str = "models/ensemble.pkl"):
        return joblib.load(path)
```

### `src/utils.py`

```python
from loguru import logger
import sys

def setup_logging(level="INFO", to_file=True, rotate="10 MB", retention="7 days"):
    logger.remove()
    logger.add(sys.stderr, level=level)
    if to_file:
        logger.add("logs/bot.log", level=level, rotation=rotate, retention=retention)
```

---

## `main.py` — live loop (demo)

```python
from __future__ import annotations
import os
import time
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.config import Cfg
from src.mt5_client import MT5Client
from src.data import fetch_bars, merge_features_labels
from src.features import build_features, FeatureConfig
from src.labels import binary_up_down
from src.ensemble import Ensemble
from src.risk import RiskManager
from src.execution import Execution
from src.utils import setup_logging

load_dotenv()

def run():
    cfg = Cfg.from_yaml("config.yaml")
    setup_logging(**cfg.logging)

    mt5c = MT5Client(os.getenv("MT5_LOGIN"), os.getenv("MT5_PASSWORD"), os.getenv("MT5_SERVER"), os.getenv("MT5_PATH") or None)
    if not mt5c.connect():
        return

    feat_cfg = FeatureConfig(**cfg.features.__dict__)
    risk = RiskManager(cfg.risk)
    exe = Execution()

    history_cache = {}
    ens_per_symbol = {}
    bars_seen = {s: 0 for s in cfg.symbols}

    while True:
        for sym in cfg.symbols:
            try:
                df = fetch_bars(sym, cfg.timeframe, cfg.history_bars)
                X = build_features(df, feat_cfg)
                y = binary_up_down(df, cfg.prediction_horizon)
                data = merge_features_labels(df, X, y)

                if sym not in ens_per_symbol or bars_seen[sym] == 0 or bars_seen[sym] % cfg.retrain_every_bars == 0:
                    logger.info(f"[{sym}] training ensemble...")
                    ens = Ensemble(cfg)
                    ens.fit(data.drop(columns=["y","close","high","low","volume"]), data["y"])
                    ens_per_symbol[sym] = ens

                ens = ens_per_symbol[sym]
                p = ens.predict_proba(X.iloc[[-1]])[-1]
                atr = X["atr_14"].iloc[-1]
                last = df["close"].iloc[-1]

                # TODO: pull real equity and pip values from MT5 account info/specs
                equity = 10000.0
                pip_value = 10.0
                pip_size = 1.0

                # Decide
                if p >= cfg.risk.min_prob_long:
                    lots = risk.position_size(equity, atr, pip_value, pip_size)
                    sl, tp = risk.stop_targets(last, atr, "long")
                    if lots > 0:
                        exe.market_order(sym, lots, "long", sl, tp)
                elif (1 - p) >= cfg.risk.min_prob_short:
                    lots = risk.position_size(equity, atr, pip_value, pip_size)
                    sl, tp = risk.stop_targets(last, atr, "short")
                    if lots > 0:
                        exe.market_order(sym, lots, "short", sl, tp)
                else:
                    logger.info(f"[{sym}] no trade. p_up={p:.3f}")

                bars_seen[sym] += 1
            except Exception as e:
                logger.exception(f"[{sym}] error: {e}")
        time.sleep(10)  # poll interval (seconds)

if __name__ == "__main__":
    run()
```

---

## Backtesting (quick start)

```python
# example usage in a notebook/script
from src.config import Cfg
from src.data import fetch_bars, merge_features_labels
from src.features import build_features, FeatureConfig
from src.labels import binary_up_down
from src.ensemble import Ensemble
from src.backtest import Backtester

cfg = Cfg.from_yaml("config.yaml")
df = fetch_bars("EURUSD", cfg.timeframe, 5000)
X = build_features(df, FeatureConfig(**cfg.features.__dict__))
y = binary_up_down(df, cfg.prediction_horizon)
ens = Ensemble(cfg)
bt = Backtester(ens, None, cfg.prediction_horizon)
curve = bt.walk_forward(df, X, y, step=250, start=1000)
print(curve.tail())
```

---

## Notes & Recommendations

* **Model diversity:** keep tree models (LGBM/XGB/RF) + a linear online learner (SGD) for regime shifts.
* **Calibration:** Isotonic calibration improves thresholding for SL/TP decisions.
* **Feature drift:** retrain frequently (e.g., every 250–500 bars) and monitor `cv_auc_`.
* **Live safety:** start with micro-lots (0.01), widen SL/TP during high-vol regimes (ATR-based).
* **Next upgrades:**

  1. Add **Optuna** tuning per symbol offline, store per-symbol params.
  2. Replace naive pip/equity with live MT5 `account_info()` and `symbol_info()` for correct lot sizing.
  3. Add **trade management** (move SL to BE at 1R, ATR trailing).
  4. Plug in real **news/sentiment** for `sentiment_score`.
  5. Persist **feature scalers** & **versioned models** in `models/`.

---

## README.md (short)

```md
# XM–MT5 AI/ML Trading Bot (Demo)

1. Create and fund an XM **demo** account.
2. Fill `.env` with login/server.
3. Edit `config.yaml` for symbols/timeframe.
4. `pip install -r requirements.txt`
5. `python main.py` (runs a simple live loop on demo).

**Backtest:** see `src/backtest.py` example.

**WARNING:** For demo use only. Markets are risky.
```













<!-- SEPTEMBER 5, 2025 | 02:10 AM -->

🔄 MT5 ML Bot Workflow
┌──────────────┐
│ MT5 Client   │
│ (login, API) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Fetch Bars   │  ← OHLCV data (history_bars)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Features     │  ← RSI, EMA, MACD, ATR, Fractals,
│ (X)          │     volatility, time cycles, etc.
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Labels (y)   │  ← Up/Down in horizon bars
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ Ensemble Training   │
│ - LGBM, XGB, RF,    │
│   Logistic Reg.     │
│ - Prob calibration  │
│ - Stacking / Voting │
└──────┬──────────────┘
       │
       ▼
┌──────────────┐
│ Prediction   │
│ p_up (0..1)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Risk Management       │
│ - Lot sizing (equity) │
│ - SL/TP (ATR-based)   │
│ - Max positions       │
│ - DD/session filter   │
└──────┬───────────────┘
       │
       ▼
┌──────────────┐
│ Trade Decision│
│ - If p_up >   │
│   min_long → BUY
│ - If (1-p_up) │
│   > min_short → SELL
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Execution    │
│ - Send order │
│ - Log result │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Trade Mgmt   │
│ - Breakeven  │
│ - Trailing SL│
└──────────────┘

🧠 How it loops:
1. Every new bar → fetch data
2. Update features & labels
3. Retrain ensemble (every retrain_every_bars)
4. Predict next move
5. Risk manager approves lot size, SL/TP
6. If conditions met → execute trade
7. Manage open trades (SL/TP adjustments)
8. Wait for next bar → repeat



🧠 Feedback on Your MT5 ML Bot
✅ Strengths
1. Walk-forward ML pipeline
	- Uses retraining (retrain_every_bars) to adapt → avoids overfitting on stale data.
	- Probabilistic ensemble (stacking/voting) is a modern approach.
2. Good feature set
	- Mix of momentum (RSI, ROC), trend (EMA), volatility (ATR), and volume.
	- Time-based bar features → gives both short-term and long-term signals.
3. Risk management built-in
	- Position sizing by equity (risk_per_trade).
	- ATR-based SL/TP and trailing stops.
	- Session filters and drawdown protection.
		→ This is better than many ML bots that just YOLO trades.
4. Logging system
	- Structured loguru logging, with INFO/ERROR separation.
	- Rotation + retention in config = production-ready.
5. Parameter tuning (Optuna)
	- Automated hyperparameter optimization per symbol.
	- Saves best params → future runs can be symbol-specific.
	- This is cutting-edge for retail trading bots.

⚠️ Weaknesses / Risks
1. Execution timing
	- Works on closed bars, not ticks → signals may be delayed, entry prices worse.
	- If retraining takes long (minutes on weak hardware), the “freshness” of predictions is lost.
2. Model stability
	- Ensemble may overfit if history_bars too small.
	- Cross-validation is there, but financial data is very noisy → AUC 
		~0.5–0.6 might not be tradable.
3. Scikit-learn warning (feature mismatch)
	- Could cause subtle bugs later if ignored.
4. SL/TP update spam
	- Bot repeatedly tries to update unchanged SL/TP, spamming logs.
	- Fix is easy but without it, execution looks messy.
5. Hardware limitations
	- Retraining with LightGBM/XGB/Optuna is heavy.
	- On a laptop, you’ll see delays (like you already noticed).
	- In live trading, this could cause missed bars or lagging trades.

🚀 Suggested Upgrades
1. Async retraining
	- Let trading loop continue while retraining runs in background thread.
	- Use last trained model until new one is ready.
2. Feature engineering
	- Add candle patterns (engulfing, pin bar).
	- Add higher timeframe features (e.g., H1 RSI while trading M5).
	- Add market microstructure features (spread, tick volume delta).
3. Smarter trade management
	- Pre-check SL/TP changes before sending updates.
	- Add “scale-in / scale-out” logic (split orders).
4. Model explainability
	- Log feature importance per retrain.
	- Helps debug why ensemble predicts long vs short.
5. Deployment readiness
	- Run bot with Docker for stability.
	- Add Prometheus/Grafana metrics for live monitoring.

🏆 My Verdict
👉 Your bot is already more advanced than 90% of “MT5 ML bots” floating around (most are just RSI+randomforest).
- Strong foundation: ML ensemble + risk management + config-driven.
- Main issue: speed and robustness on live trading.
- Next level: optimize execution speed, enrich features, and make it production-grade with monitoring.

<!-- SEPTEMBER 5, 2025 | 02:10 AM -->