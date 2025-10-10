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

def add_contextual_features(df: pd.DataFrame, mta_df: pd.DataFrame = None, inter_market_df: pd.DataFrame = None, mta_cfg: "MtaCfg" = None, im_cfg: "InterMarketCfg" = None) -> pd.DataFrame:
    """
    Adds contextual features from higher timeframes (MTA) and other markets.
    """
    if mta_df is not None and mta_cfg and mta_cfg.enabled:
        logger.debug(f"Adding MTA features from timeframe {mta_cfg.timeframe}...")
        # Calculate indicators on MTA dataframe
        mta_ema = ta.trend.ema_indicator(mta_df["close"], window=mta_cfg.ema_period)
        mta_rsi = ta.momentum.rsi(mta_df["close"], window=mta_cfg.rsi_period)

        # Create a dataframe for these features
        mta_features = pd.DataFrame(index=mta_df.index)
        mta_features[f'mta_ema_{mta_cfg.ema_period}'] = mta_ema
        mta_features[f'mta_rsi_{mta_cfg.rsi_period}'] = mta_rsi

        # Align with the primary dataframe
        df = pd.merge(df, mta_features, left_index=True, right_index=True, how='left')
        df.ffill(inplace=True)

    if inter_market_df is not None and im_cfg and im_cfg.enabled:
        logger.debug(f"Adding Inter-Market features from symbol {im_cfg.symbol}...")
        im_features = pd.DataFrame(index=inter_market_df.index)
        for lag in im_cfg.roc_lags:
            im_features[f'im_{im_cfg.symbol}_roc_{lag}'] = inter_market_df["close"].pct_change(lag)

        # Align with the primary dataframe
        df = pd.merge(df, im_features, left_index=True, right_index=True, how='left')
        df.ffill(inplace=True)

    return df

def build_static_features(df: pd.DataFrame, symbol: str = None, pa_cfg: "PriceActionCfg" = None) -> pd.DataFrame:
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

    # --- Advanced Price Action Features ---
    if pa_cfg and pa_cfg.enabled:
        # Distance from 'Home Base' MA
        home_base_ma = ta.trend.ema_indicator(df["close"], window=pa_cfg.home_base_ma_period)
        X[f'dist_from_ema_{pa_cfg.home_base_ma_period}'] = (df["close"] - home_base_ma) / home_base_ma

        # Time since N-bar high/low
        rolling_high = df["high"].rolling(window=pa_cfg.swing_lookback).max()
        rolling_low = df["low"].rolling(window=pa_cfg.swing_lookback).min()
        
        is_new_high = df["high"] == rolling_high
        is_new_low = df["low"] == rolling_low

        # Cumulatively count bars since the last event
        X['bars_since_high'] = is_new_high.cumsum().groupby((is_new_high).cumsum()).cumcount()
        X['bars_since_low'] = is_new_low.cumsum().groupby((is_new_low).cumsum()).cumcount()

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

from src.config import Cfg # Import Cfg

def build_features(df: pd.DataFrame, feature_cfg: FeatureConfig, main_cfg: Cfg, symbol: str = None, mta_df: pd.DataFrame = None, inter_market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Original build_features function, now delegates to static and dynamic builders.
    This remains for compatibility with other scripts that may use it directly.
    """
    pa_cfg = getattr(getattr(main_cfg, 'context_features', None), 'price_action', None)
    mta_cfg = getattr(main_cfg.context_features, 'mta', None)
    im_cfg = getattr(main_cfg.context_features, 'inter_market', None)

    static_X = build_static_features(df, symbol, pa_cfg=pa_cfg)
    dynamic_X = build_dynamic_features(df, static_X, feature_cfg, symbol)
    
    # Add contextual features
    dynamic_X = add_contextual_features(dynamic_X, mta_df=mta_df, inter_market_df=inter_market_df, mta_cfg=mta_cfg, im_cfg=im_cfg)
    
    return dynamic_X

def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = (fwd > 0).astype(int)
    return y.loc[df.index]