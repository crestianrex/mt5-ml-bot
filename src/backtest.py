### `src/backtest.py`

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
