### `src/strategy_ml.py` (extended to keep your API and add calibration)

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
from loguru import logger

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
                verbose=-1,  # <-- add this line to suppress LightGBM warnings
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
            self._calibrator = CalibratedClassifierCV(estimator=self._pipe.named_steps["clf"], cv=None, method="isotonic")
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
            logger.warning("Online update skipped: empty new data")
            return
        if self.supports_online:
            self._pipe.named_steps["clf"].partial_fit(Xn, yn, classes=[0,1])
            logger.info(f"Online update performed: {len(Xn)} samples")
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
            logger.info(f"Offline retrain performed: total samples={len(Xc)}")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        Xc = self._sanitize(X)
        if len(Xc) == 0:
            return pd.Series(0.5, index=X.index, name="p_up")
        if self.calibrate and self._calibrator is not None:
            p = self._calibrator.predict_proba(Xc)[:,1]
        else:
            p = self._proba_raw(Xc)
        return pd.Series(p, index=Xc.index, name="p_up")
