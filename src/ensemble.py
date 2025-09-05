### src/ensemble.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
from .strategy_ml import MLStrategy
from sklearn.isotonic import IsotonicRegression
from loguru import logger

class Ensemble:
    def __init__(self, cfg, model_params: dict | None = None):
        """
        cfg: configuration object
        model_params: optional per-symbol Optuna-tuned params, dict like:
            {
                "lgbm": {...},
                "xgb": {...},
                "rf": {...},
                "logreg": {...}
            }
        """
        self.cfg = cfg
        self.members: Dict[str, MLStrategy] = {}
        for m in cfg.models:
            name = m["name"]
            params = m.get("params", {}).copy()
            if model_params and name in model_params:
                params.update(model_params[name])
            # Force calibrate=True inside MLStrategy for per-model isotonic
            self.members[name] = MLStrategy(model=name, calibrate=True, **params)

        self.method = cfg.ensemble.get("method", "soft_vote")
        self.weights = cfg.ensemble.get("weights", {k: 1/len(self.members) for k in self.members})
        self.meta = cfg.ensemble.get("meta", {"type": "logit", "C": 1.0})
        self._stacker = None
        self._meta_calibrator: IsotonicRegression | None = None

    def update_params(self, new_params: dict):
        """
        Update base learner parameters dynamically.

        new_params: dict like
        {
            "lgbm": {"n_estimators":400, "learning_rate":0.03, ...},
            "xgb": {...},
            "rf": {...},
            "logreg": {}
        }
        """
        if not new_params:
            logger.warning("No new parameters provided to update.")
            return

        for name, params in new_params.items():
            if name in self.members and params:
                self.members[name].set_params(**params)
                logger.info(f"[{name}] Parameters updated: {params}")
                self.model_params[name] = params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logger.info("Fitting ensemble members...")
        """
        Fit all member models and optionally fit stacking meta-model
        """
        Ps = []
        for name, model in self.members.items():
            model.fit(X, y)
            p = model.predict_proba(X).rename(name)  # already calibrated
            Ps.append(p)
            logger.info(f"[{name}] Model fitted, CV AUC={getattr(model, 'cv_auc_', None):.4f}")

        P = pd.concat(Ps, axis=1)

        if self.method == "stacking":
            from sklearn.linear_model import LogisticRegression
            self._stacker = LogisticRegression(C=self.meta.get("C", 1.0), max_iter=200)
            self._stacker.fit(P.values, y.loc[P.index].values)
            logger.info("Stacking meta-model fitted.")

            # Optional isotonic calibration for stacking output
            p_meta = self._stacker.predict_proba(P.values)[:, 1]
            self._meta_calibrator = IsotonicRegression(out_of_bounds='clip')
            self._meta_calibrator.fit(p_meta, y.loc[P.index])
            logger.info("Meta-model isotonic calibration complete.")

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Return calibrated probability of "up" for ensemble
        """
        Pcols = [m.predict_proba(X).rename(n) for n, m in self.members.items()]
        P = pd.concat(Pcols, axis=1)
        logger.debug(f"Ensemble predicting: input shape={X.shape}, member probs shape={P.shape}")

        if self.method == "soft_vote":
            w = np.array([self.weights.get(k, 1.0) for k in P.columns], dtype=float)
            w /= w.sum()
            p_final = (P.values * w).sum(axis=1)

        elif self.method == "stacking" and self._stacker is not None:
            p_final = self._stacker.predict_proba(P.values)[:, 1]
            if self._meta_calibrator is not None:
                p_final = self._meta_calibrator.transform(p_final)

        elif self.method == "risk_weighted":
            eps = 1e-9
            mus = P.rolling(200).mean()
            sig = P.rolling(200).std()
            score = (mus / (sig + eps)).iloc[-1].fillna(0.0)
            w = (score.clip(lower=0) + eps).values
            if w.sum() == 0:
                w = np.ones_like(w)
            w /= w.sum()
            p_final = (P.values * w).sum(axis=1)

        else:
            p_final = P.mean(axis=1).values

        return pd.Series(p_final, index=P.index, name="p_up")

    
