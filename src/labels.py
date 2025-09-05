### `src/labels.py`

import pandas as pd

def binary_up_down(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int)
