# src/features.py
import pandas as pd
import numpy as np
import ta
from loguru import logger

class FeatureConfig:
    def __init__(self, rsi_period=14, ema_fast=12, ema_slow=26, window_vol=20, roc_lags=(1,3,5,10), adx_period=14, rsi_ob_level=70, rsi_os_level=30, adx_trend_thresh=25, timeframe_minutes=5):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.window_vol = window_vol
        self.roc_lags = list(roc_lags)
        self.adx_period = adx_period
        self.rsi_ob_level = rsi_ob_level
        self.rsi_os_level = rsi_os_level
        self.adx_trend_thresh = adx_trend_thresh
        self.timeframe_minutes = timeframe_minutes

def build_static_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Builds features that do not depend on tunable hyperparameters.
    These can be calculated once and cached.
    """
    logger.debug(f"[{symbol}] Building static features...")
    X = pd.DataFrame(index=df.index)
    
    # --- MACD ---
    macd = ta.trend.MACD(df["close"])
    X["macd"] = macd.macd()
    X["macd_signal"] = macd.macd_signal()
    X["macd_diff"] = macd.macd_diff()

    # --- Fixed Momentum ---
    X["momentum_5"] = df["close"] - df["close"].shift(5)
    X["momentum_10"] = df["close"] - df["close"].shift(10)

    # --- Fixed Volatility ---
    X["atr_14"] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    X["volatility_10"] = df["close"].pct_change().rolling(10).std()
    X["volatility_20"] = df["close"].pct_change().rolling(20).std()

    # --- Volume Features (if available) ---
    if "volume" in df.columns:
        X["vol_ma_20"] = df["volume"].rolling(20).mean()
        X["vol_ratio"] = df["volume"] / (X["vol_ma_20"] + 1e-10)

    # --- Fractal Features ---
    X["fractal_up"] = ((df["high"].shift(2) < df["high"].shift(1)) & (df["high"].shift(1) > df["high"]) & (df["high"].shift(1) > df["high"].shift(-1)) & (df["high"].shift(1) > df["high"].shift(-2))).astype(int)
    X["fractal_down"] = ((df["low"].shift(2) > df["low"].shift(1)) & (df["low"].shift(1) < df["low"]) & (df["low"].shift(1) < df["low"].shift(-1)) & (df["low"].shift(1) < df["low"].shift(-2))).astype(int)

    # --- Rolling Statistics ---
    X["ret_skew_10"] = df["close"].pct_change().rolling(10).skew()
    X["ret_kurt_10"] = df["close"].pct_change().rolling(10).kurt()

    # --- Time-based Features ---
    X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    return X

def build_dynamic_features(df: pd.DataFrame, static_features: pd.DataFrame, cfg: FeatureConfig, symbol: str = None) -> pd.DataFrame:
    """
    Builds features that depend on tunable hyperparameters, using pre-calculated static features.
    """
    X = static_features.copy()

    try:
        # --- Price/Momentum Features (Dynamic) ---
        X["rsi"] = ta.momentum.rsi(df["close"], window=cfg.rsi_period)
        X["ema_fast"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_fast)
        X["ema_slow"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_slow)
        X["ema_diff"] = (X["ema_fast"] - X["ema_slow"]) / df["close"]

        for l in cfg.roc_lags:
            X[f"ret_{l}"] = df["close"].pct_change(l)

        # --- Volatility Features (Dynamic) ---
        bb = ta.volatility.BollingerBands(df["close"], window=cfg.window_vol, window_dev=2)
        X["bb_high"] = bb.bollinger_hband()
        X["bb_low"] = bb.bollinger_lband()
        X["bb_width"] = (X["bb_high"] - X["bb_low"]) / df["close"]
        
        # --- Regime Detection & Mean-Reversion Features (Dynamic) ---
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=cfg.adx_period)
        X["adx"] = adx_indicator.adx()
        X["is_trending"] = (X["adx"] > cfg.adx_trend_thresh).astype(int)

        X["rsi_ob"] = (X["rsi"] > cfg.rsi_ob_level).astype(int)
        X["rsi_os"] = (X["rsi"] < cfg.rsi_os_level).astype(int)

        X["bb_touch_upper"] = (df["high"] >= X["bb_high"]).astype(int)
        X["bb_touch_lower"] = (df["low"] <= X["bb_low"]).astype(int)

        # --- handle NaNs and infs ---
        nan_count = X.isna().sum().sum()
        inf_count = np.isinf(X.values).sum()
        X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        logger.debug(f"[{symbol}] Dynamic features built. Shape={X.shape}, NaNs filled={nan_count}, Infs replaced={inf_count}")

    except Exception as e:
        logger.exception(f"[{symbol}] Error building dynamic features: {e}")
        raise

    return X

def build_features(df: pd.DataFrame, cfg: FeatureConfig, symbol: str = None, timeframe_minutes: int = 5) -> pd.DataFrame:
    """
    Original build_features function, now delegates to static and dynamic builders.
    This remains for compatibility with other scripts that may use it directly.
    """
    static_X = build_static_features(df, symbol)
    dynamic_X = build_dynamic_features(df, static_X, cfg, symbol)
    return dynamic_X

def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = (fwd > 0).astype(int)
    return y.loc[df.index]