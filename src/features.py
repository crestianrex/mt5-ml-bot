### src/features.py — with recency weighting + symbol filtering

import pandas as pd
import numpy as np
import ta
from loguru import logger
# from newsapi import NewsApiClient
from datetime import datetime, timedelta
from dateutil import parser  # pip install python-dateutil
# import nltk
# nltk.download("vader_lexicon")  # Only needs to run once
# from nltk.sentiment import SentimentIntensityAnalyzer
import time
# from requests.exceptions import SSLError

# Initialize clients
# newsapi = NewsApiClient(api_key="396c29fd806f4c6aa1afe1af7e094de6")
# sia = SentimentIntensityAnalyzer()

DEF_FILL = {"method": "ffill"}

class FeatureConfig:
    def __init__(self, rsi_period=14, ema_fast=12, ema_slow=26, window_vol=20, roc_lags=(1,3,5,10), timeframe_minutes=5):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.window_vol = window_vol
        self.roc_lags = list(roc_lags)
        self.timeframe_minutes = timeframe_minutes

# # --- Step 1: Fetch news ---
# def fetch_news(symbol: str, timeframe_minutes: int = 5, retries=3) -> list[dict]:
#     """
#     Fetch latest news articles for a Forex symbol.
#     Returns a list of article dicts (title + publishedAt, etc.).
#     """
#     now = datetime.utcnow()
#     from_time = (now - timedelta(minutes=timeframe_minutes)).replace(microsecond=0)
#     to_time = now.replace(microsecond=0)

#     for attempt in range(retries):
#         try:
#             all_articles = newsapi.get_everything(
#                 q=symbol,
#                 from_param=from_time.isoformat(),
#                 to=to_time.isoformat(),
#                 language='en',
#                 sort_by='publishedAt',
#                 page_size=20
#             )
#             return [a for a in all_articles['articles']]  # full dict for recency weighting
#         except SSLError as e:
#             print(f"[NewsAPI] SSL error, retrying {attempt+1}/{retries}...")
#             time.sleep(2)
#         except Exception as e:
#             print(f"[NewsAPI] Error fetching news: {e}")
#             return []
#     return []

# # --- Step 2: Compute recency-weighted sentiment ---
# def compute_sentiment(headlines: list[dict], recency_half_life: float = None) -> float:
#     """
#     Convert list of headlines into a numeric sentiment score.
#     Applies exponential decay weighting based on recency.
#     recency_half_life in seconds (default fallback 5 min)
#     """
#     if not headlines:
#         return 0.0

#     if recency_half_life is None:
#         recency_half_life = 300.0  # default fallback if not provided

#     now = datetime.utcnow()
#     scores = []
#     weights = []

#     for h in headlines:
#         title = h["title"]
#         pub_time = parser.isoparse(h["publishedAt"])  # string -> datetime
#         delta_sec = (now - pub_time).total_seconds()
#         weight = 0.5 ** (delta_sec / recency_half_life)
#         score = sia.polarity_scores(title)["compound"]
#         scores.append(score * weight)
#         weights.append(weight)

#     return sum(scores) / (sum(weights) + 1e-10)

# --- Step 3: Build features with sentiment integration ---
def build_features(df: pd.DataFrame, cfg: FeatureConfig, symbol: str = None, timeframe_minutes: int = 5) -> pd.DataFrame:

    X = pd.DataFrame(index=df.index)
    try:

        # --- price/momentum features ---
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

        # # --- news/sentiment caching per symbol ---
        # cache = getattr(build_features, "_news_cache", {})
        # tf_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        # timeframe_minutes = tf_map.get(cfg.timeframe, 5)  # fallback 5 min
        # news_cache_interval = timeframe_minutes * 60

        # if symbol in cache and (datetime.utcnow() - cache[symbol]["time"]).total_seconds() < news_cache_interval:
        #     headlines = cache[symbol]["headlines"]
        # else:
        #     try:
        #         headlines = fetch_news(symbol, timeframe_minutes=5)
        #     except Exception as e:
        #         print(f"Error fetching news for {symbol}: {e}")
        #         headlines = []
        #     cache[symbol] = {"headlines": headlines, "time": datetime.utcnow()}
        # setattr(build_features, "_news_cache", cache)

        # # --- filter for exact symbol match ---
        # headlines = [h for h in headlines if symbol in h["title"]]

        # --- compute weighted sentiment ---
        X["sentiment_score"] = 0.0
        # X["sentiment_score"] = compute_sentiment(headlines, recency_half_life=timeframe_minutes * 60)

        X["ret_skew_10"] = df["close"].pct_change().rolling(10).skew()
        X["ret_kurt_10"] = df["close"].pct_change().rolling(10).kurt()

        X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        X["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        X["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # --- handle NaNs and infs ---
        nan_count = X.isna().sum().sum()
        inf_count = np.isinf(X.values).sum()
        X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        logger.debug(f"[{symbol}] Features built. Shape={X.shape}, NaNs filled={nan_count}, Infs replaced={inf_count}")
    
    except Exception as e:
        logger.exception(f"[{symbol}] Error building features: {e}")
        raise

    return X

def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = (fwd > 0).astype(int)
    return y.loc[df.index]
