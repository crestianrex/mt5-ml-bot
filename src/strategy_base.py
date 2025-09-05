### `src/strategy_base.py`

from __future__ import annotations
import pandas as pd

class Strategy:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def online_update(self, X_new: pd.DataFrame, y_new: pd.Series, X_hist=None, y_hist=None):
        pass
