#src/data_manager.py
from __future__ import annotations
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Tuple
import datetime

from src.config import Cfg
from src.features import FeatureConfig, build_features, make_labels

def ensure_dir(path: str):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)

class DataManager:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.raw_data_dir = cfg.fetch.raw_data_dir
        ensure_dir(self.raw_data_dir)

    def _local_csv_path(self, symbol: str, timeframe: str) -> str:
        fname = f"{symbol.replace('#','')}_{timeframe}.csv"
        return os.path.join(self.raw_data_dir, fname)

    def _atomic_write_df(self, df: pd.DataFrame, path: str, fmt: str = "csv"):
        ensure_dir(os.path.dirname(path))
        fd, tmp = tempfile.mkstemp(prefix="tmp_", dir=os.path.dirname(path))
        os.close(fd)
        try:
            if fmt == "csv":
                df.to_csv(tmp, index=True)
            else:
                df.to_parquet(tmp, index=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def load_local_history(self, symbol: str, timeframe: str, count: Optional[int] = None) -> pd.DataFrame:
        path = self._local_csv_path(symbol, timeframe)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0)
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        if count is not None and len(df) > count:
            df = df.tail(count)
        return df

    def append_new_bars(self, symbol: str, new_bars: pd.DataFrame):
        if not isinstance(new_bars, pd.DataFrame) or new_bars.empty:
            logger.debug(f"[{symbol}] No new bars to append.")
            return
        path = self._local_csv_path(symbol, self.cfg.timeframe)
        nb = new_bars.copy()
        try:
            nb.index = pd.to_datetime(nb.index)
        except Exception:
            nb.index = pd.to_datetime(nb.index.astype(str))

        if os.path.exists(path):
            existing = pd.read_csv(path, index_col=0)
            try:
                existing.index = pd.to_datetime(existing.index)
            except Exception:
                pass
            combined = pd.concat([existing, nb])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        else:
            combined = nb.sort_index()

        self._atomic_write_df(combined, path, fmt="csv")
        logger.debug(f"[{symbol}] Appended {len(nb)} bars to {path} (total {len(combined)})")

    def _fetch_bars_from_mt5_chunked(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        import MetaTrader5 as mt5  # type: ignore
        TF_MAP = {
            "M1": getattr(mt5, "TIMEFRAME_M1", None),
            "M5": getattr(mt5, "TIMEFRAME_M5", None),
            "M15": getattr(mt5, "TIMEFRAME_M15", None),
            "M30": getattr(mt5, "TIMEFRAME_M30", None),
            "H1": getattr(mt5, "TIMEFRAME_H1", None),
            "H4": getattr(mt5, "TIMEFRAME_H4", None),
            "D1": getattr(mt5, "TIMEFRAME_D1", None),
        }
        tf = TF_MAP.get(str(timeframe).upper())
        if tf is None:
            logger.error(f"[{symbol}] Unsupported timeframe: {timeframe}")
            return pd.DataFrame()

        if count is None:
            count = 36000
        try:
            logger.debug(f"[{symbol}] Requesting {count} bars of {timeframe} from MT5 (tf={tf})...")
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
            logger.debug(f"[{symbol}] MT5 copy_rates_from_pos returned: {rates[:5] if rates is not None else None}...") # Log first 5 rates for brevity
            if rates is None or len(rates) == 0:
                logger.warning(f"[{symbol}] MT5 returned no bars.")
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            if "time" not in df.columns:
                logger.warning(f"[{symbol}] fetched data missing 'time' column — returning empty DataFrame")
                return pd.DataFrame()
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("time").sort_index()
            df = df.rename(columns={"tick_volume": "volume"})
            return df[["open","high","low","close","volume"]].copy()
        except Exception as e:
            logger.exception(f"[{symbol}] Error fetching bars: {e}")
            return pd.DataFrame()

    def bootstrap_history(self, symbol: str, initial_bars: int):
        path = self._local_csv_path(symbol, self.cfg.timeframe)
        current = self.load_local_history(symbol, self.cfg.timeframe)
        if current.empty or len(current) < initial_bars:
            logger.info(f"[{symbol}] Bootstrapping local history: have={len(current)}, need={initial_bars}")
            df = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.timeframe, initial_bars)
            if df.empty:
                logger.warning(f"[{symbol}] Bootstrap failed: MT5 returned no data.")
                return
            self._atomic_write_df(df, path, fmt="csv")
            logger.info(f"[{symbol}] Bootstrapped local history to {path} ({len(df)} rows).")
        else:
            logger.debug(f"[{symbol}] Local history OK ({len(current)} rows).")

    def fetch_live(self, symbol: str, feature_cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # 1. Fetch latest N bars for feature calculation lookback
        # Use history_bars as a safe lookback for feature calculation
        lookback_bars = self.cfg.history_bars 

        if self.cfg.data_source == "csv":
            data = self.load_local_history(symbol, self.cfg.timeframe, count=lookback_bars)
        elif self.cfg.data_source == "mt5":
            data = self._fetch_bars_from_mt5_chunked(symbol, self.cfg.timeframe, lookback_bars)
        else:
            raise ValueError(f"Unknown data source: {self.cfg.data_source}")

        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        X, y = self._build_features_and_labels(data, feature_cfg, symbol)
        return data, X, y

    def load_cached(self, symbol: str, feature_cfg: FeatureConfig, full: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        count = None if full else self.cfg.history_bars
        data = self.load_local_history(symbol, self.cfg.timeframe, count=count)
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        X, y = self._build_features_and_labels(data, feature_cfg, symbol)
        return data, X, y

    def _build_features_and_labels(self, data: pd.DataFrame, feature_cfg: FeatureConfig, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # This can be a shared method for both live and cached data loading
        mta_df = None
        inter_market_df = None

        # Fetch MTA data if enabled
        if self.cfg.context_features.mta.enabled:
            mta_timeframe = self.cfg.context_features.mta.timeframe
            # Fetch enough bars for MTA lookback (e.g., ema_period, rsi_period)
            mta_lookback = max(self.cfg.context_features.mta.ema_period, self.cfg.context_features.mta.rsi_period) * 2 # A bit extra
            
            if self.cfg.data_source == "csv":
                mta_df = self.load_local_history(symbol, mta_timeframe, count=mta_lookback)
            elif self.cfg.data_source == "mt5":
                mta_df = self._fetch_bars_from_mt5_chunked(symbol, mta_timeframe, mta_lookback)
            else:
                raise ValueError(f"Unknown data source: {self.cfg.data_source}")

            if mta_df.empty:
                logger.warning(f"[{symbol}] No MTA data fetched for timeframe {mta_timeframe}.")
                mta_df = None

        # Fetch Inter-Market data if enabled
        if self.cfg.context_features.inter_market.enabled:
            im_symbol = self.cfg.context_features.inter_market.symbol
            im_lags = max(self.cfg.context_features.inter_market.roc_lags) if self.cfg.context_features.inter_market.roc_lags else 1
            im_lookback = im_lags * 2 # A bit extra
            
            if self.cfg.data_source == "csv":
                inter_market_df = self.load_local_history(im_symbol, self.cfg.timeframe, count=im_lookback)
            elif self.cfg.data_source == "mt5":
                inter_market_df = self._fetch_bars_from_mt5_chunked(im_symbol, self.cfg.timeframe, im_lookback)
            else:
                raise ValueError(f"Unknown data source: {self.cfg.data_source}")

            if inter_market_df.empty:
                logger.warning(f"[{symbol}] No Inter-Market data fetched for symbol {im_symbol}.")
                inter_market_df = None

        X = build_features(data.copy(), feature_cfg, self.cfg, symbol)
        y = make_labels(data.copy(), self.cfg.prediction_horizon)

        # Align X and y by index
        aligned_idx = X.index.intersection(y.index)
        X = X.loc[aligned_idx]
        y = y.loc[aligned_idx]

        return X, y