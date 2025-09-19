# src/strategy_ml.py
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from loguru import logger

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# Min samples required to safely train a model and get a reliable CV score
MIN_SAMPLES_FOR_FIT = 150


class MLStrategy:
    def __init__(self, model="lgbm", random_state: int = 42, calibrate: bool = True, cv_samples_per_split: int = 300, **kwargs):
        self.model_name = model.lower()
        self.random_state = random_state
        self.calibrate = bool(calibrate)
        self.cv_samples_per_split = int(cv_samples_per_split)
        model_params = kwargs.copy()
        device = model_params.pop("device", "cpu")
        self._calibrator = None

        # seed numpy for reproducibility of any random ops here
        np.random.seed(self.random_state)

        if self.model_name == "rf":
            base = RandomForestClassifier(
                n_estimators=int(model_params.get("n_estimators", 200)),
                max_depth=model_params.get("max_depth", None),
                min_samples_leaf=model_params.get("min_samples_leaf", 3),
                n_jobs=-1,
                random_state=self.random_state,
            )
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])

        elif self.model_name == "xgb":
            if XGBClassifier is None:
                raise ImportError("xgboost not installed")
            xgb_params = {
                "n_estimators": int(model_params.get("n_estimators", 200)),
                "max_depth": int(model_params.get("max_depth", 5)),
                "learning_rate": float(model_params.get("learning_rate", 0.05)),
                "subsample": float(model_params.get("subsample", 0.8)),
                "colsample_bytree": float(model_params.get("colsample_bytree", 0.8)),
                "random_state": self.random_state,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
            base = XGBClassifier(**xgb_params)
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])

        elif self.model_name == "lgbm":
            if LGBMClassifier is None:
                raise ImportError("lightgbm not installed")
            lgbm_params = {
                "n_estimators": int(model_params.get("n_estimators", 200)),
                "max_depth": int(model_params.get("max_depth", -1)),
                "learning_rate": float(model_params.get("learning_rate", 0.05)),
                "subsample": float(model_params.get("subsample", 0.8)),
                "colsample_bytree": float(model_params.get("colsample_bytree", 0.8)),
                "min_child_samples": int(model_params.get("min_child_samples", 5)),
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            base = LGBMClassifier(**lgbm_params)
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])

        elif self.model_name == "logreg" or self.model_name == "sgd":
            # Use SGD for online-capable logistic-like model
            sgd = SGDClassifier(loss="log_loss", learning_rate="optimal", max_iter=1000, tol=1e-3, warm_start=True, random_state=self.random_state)
            self.supports_online = True
            self._pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", sgd)])

        else:
            raise ValueError(f"Unknown model '{self.model_name}'")

    def _sanitize(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        Xc = self._sanitize(X)
        yc = y.reindex(Xc.index)
        if len(Xc) < MIN_SAMPLES_FOR_FIT:
            logger.warning(f"Not enough data to fit model: {len(Xc)} < {MIN_SAMPLES_FOR_FIT}. Skipping fit.")
            self.cv_auc_ = 0.5
            return self

        n_splits = min(5, max(2, len(Xc) // self.cv_samples_per_split))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        last_tr_idx, last_va_idx = None, None

        for tr, va in tscv.split(Xc):
            if len(tr) < 20 or len(va) < 20:
                continue
            last_tr_idx, last_va_idx = tr, va
            try:
                self._pipe.fit(Xc.iloc[tr], yc.iloc[tr])
                p = self._proba_raw(Xc.iloc[va])
                scores.append(roc_auc_score(yc.iloc[va], p))
            except Exception as e:
                logger.warning(f"CV fold failed: {e}")
                continue

        self.cv_auc_ = float(np.mean(scores)) if scores else 0.5

        # Fit final model on last split (or on all data if no cv)
        try:
            if last_tr_idx is not None and last_va_idx is not None:
                self._pipe.fit(Xc.iloc[last_tr_idx], yc.iloc[last_tr_idx])
            else:
                self._pipe.fit(Xc, yc)
        except Exception as e:
            logger.exception(f"Final fit failed: {e}")
            self.cv_auc_ = 0.5
            return self

        # Try calibration if requested
        if self.calibrate:
            try:
                # Use CalibratedClassifierCV which will internally refit if cv is integer,
                # wrapping the whole pipeline. This is slower but safer and consistent.
                self._calibrator = CalibratedClassifierCV(self._pipe, cv=3, method="isotonic")
                self._calibrator.fit(Xc, yc)
                logger.info("Calibration finished")
            except Exception as e:
                logger.warning(f"Calibration failed or unsupported: {e}")
                self._calibrator = None

        return self

    def _proba_raw(self, X: pd.DataFrame):
        # Return numpy array of probabilities for the positive class (1)
        try:
            if hasattr(self._pipe, "predict_proba"):
                arr = self._pipe.predict_proba(X)[:, 1]
                return arr
            clf = self._pipe.named_steps.get("clf")
            if clf is not None and hasattr(clf, "decision_function"):
                dec = clf.decision_function(X)
                return 1.0 / (1.0 + np.exp(-dec))
            # fallback uniform
            return np.full(len(X), 0.5)
        except Exception as e:
            logger.exception(f"_proba_raw failed: {e}")
            return np.full(len(X), 0.5)

    def online_update(self, X_new: pd.DataFrame, y_new: pd.Series, X_hist: pd.DataFrame = None, y_hist: pd.Series = None):
        Xn = self._sanitize(X_new)
        yn = y_new.reindex(Xn.index)
        if len(Xn) == 0:
            logger.warning("Online update skipped: empty new data")
            return

        # If underlying classifier supports partial_fit, use that for speed
        clf = self._pipe.named_steps.get("clf")
        if hasattr(clf, "partial_fit") and self.supports_online:
            try:
                classes = [0, 1]
                clf.partial_fit(Xn, yn, classes=classes)
                logger.info(f"Partial fit performed: {len(Xn)} samples")
                # optionally refresh calibrator lightly
                if self.calibrate:
                    try:
                        # retrain calibrator on recent sample
                        self._calibrator = CalibratedClassifierCV(self._pipe, cv=3, method="isotonic")
                        self._calibrator.fit(Xn, yn)
                    except Exception:
                        logger.debug("Calibrator refresh failed during online update")
                return
            except Exception as e:
                logger.warning(f"Partial-fit failed: {e}")

        # fallback: offline retrain using concatenated history
        if X_hist is not None and y_hist is not None:
            Xc = pd.concat([X_hist, Xn]).loc[~pd.concat([X_hist, Xn]).index.duplicated(keep="last")]
            yc = pd.concat([y_hist, yn]).loc[Xc.index]
        else:
            Xc, yc = Xn, yn
        logger.info(f"Offline retrain from online_update: total samples={len(Xc)}")
        self.fit(Xc, yc)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        Xc = self._sanitize(X)
        if len(Xc) == 0:
            return pd.Series(0.5, index=X.index, name="p_up")
        try:
            if self._calibrator is not None:
                p = self._calibrator.predict_proba(Xc)[:, 1]
                return pd.Series(p, index=Xc.index, name="p_up")
            p = self._proba_raw(Xc)
            return pd.Series(p, index=Xc.index, name="p_up")
        except Exception as e:
            logger.exception(f"predict_proba failed: {e}")
            return pd.Series(0.5, index=X.index, name="p_up")
