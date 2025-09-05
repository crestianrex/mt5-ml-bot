### `src/trainer.py`

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
