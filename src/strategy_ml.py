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
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from .strategy_base import Strategy

# Min samples required to safely train a model and get a reliable CV score
MIN_SAMPLES_FOR_FIT = 150

class MLStrategy(Strategy):
    def __init__(self, model="lgbm", random_state=42, calibrate=True, cv_samples_per_split=300, **kwargs):
        self.model_name = model
        self.random_state = random_state
        self.calibrate = calibrate
        self.cv_samples_per_split = cv_samples_per_split

        model_params = kwargs.copy()
        device = model_params.pop('device', 'cpu') # Pop device, default to cpu

        if model == "rf":
            base = RandomForestClassifier(
                n_estimators=model_params.get("n_estimators", 200),
                max_depth=model_params.get("max_depth", None),
                min_samples_leaf=model_params.get("min_samples_leaf", 3),
                n_jobs=-1,
                random_state=random_state,
            )
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])

        elif model == "xgb":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed")
            
            xgb_params = {
                "n_estimators": model_params.get("n_estimators", 200),
                "max_depth": model_params.get("max_depth", 5),
                "learning_rate": model_params.get("learning_rate", 0.05),
                "subsample": model_params.get("subsample", 0.8),
                "colsample_bytree": model_params.get("colsample_bytree", 0.8),
                "random_state": random_state,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
            if device == 'gpu':
                logger.info("🔥 Configuring XGBoost for GPU...")
                xgb_params["device"] = "cuda"
                xgb_params["tree_method"] = "hist"

            base = XGBClassifier(**xgb_params)
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])

        elif model == "lgbm":
            if LGBMClassifier is None:
                raise ImportError("lightgbm is not installed")

            lgbm_params = {
                "n_estimators": model_params.get("n_estimators", 200),
                "max_depth": model_params.get("max_depth", -1),
                "learning_rate": model_params.get("learning_rate", 0.05),
                "subsample": model_params.get("subsample", 0.8),
                "colsample_bytree": model_params.get("colsample_bytree", 0.8),
                "min_child_samples": model_params.get("min_child_samples", 5),
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            if device == 'gpu':
                logger.info("🔥 Configuring LightGBM for GPU...")
                lgbm_params["device"] = "gpu"

            base = LGBMClassifier(**lgbm_params)
            self.supports_online = False
            self._pipe = Pipeline([("clf", base)])

        elif model == "logreg":
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
        
        if len(Xc) < MIN_SAMPLES_FOR_FIT:
            logger.warning(f"Not enough data to fit model: {len(Xc)} samples < {MIN_SAMPLES_FOR_FIT}. Skipping fit.")
            self.cv_auc_ = 0.5
            return self

        n_splits = min(5, max(2, len(Xc) // self.cv_samples_per_split))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        last_tr_idx, last_va_idx = None, None

        for tr, va in tscv.split(Xc):
            if len(tr) < 50 or len(va) < 50:
                continue

            last_tr_idx, last_va_idx = tr, va
            self._pipe.fit(Xc.iloc[tr], yc.iloc[tr])
            p = self._proba_raw(Xc.iloc[va])
            scores.append(roc_auc_score(yc.iloc[va], p))

        self.cv_auc_ = float(np.mean(scores)) if scores else 0.5

        if self.model_name in ["lgbm", "xgb"] and last_tr_idx is not None and last_va_idx is not None:
            X_train_final, X_val_final = Xc.iloc[last_tr_idx], Xc.iloc[last_va_idx]
            y_train_final, y_val_final = yc.iloc[last_tr_idx], yc.iloc[last_va_idx]

            clf = self._pipe.named_steps['clf']
            
            if self.model_name == "lgbm":
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
                clf.fit(X_train_final, y_train_final,
                        eval_set=[(X_val_final, y_val_final)],
                        eval_metric="auc", 
                        callbacks=callbacks)
            elif self.model_name == "xgb":
                clf.fit(X_train_final, y_train_final,
                        eval_set=[(X_val_final, y_val_final)], verbose=False)
            else:
                self._pipe.fit(Xc, yc)
        else:
            self._pipe.fit(Xc, yc)

        if self.calibrate:
            self._calibrator = CalibratedClassifierCV(estimator=self._pipe.named_steps["clf"], cv=None, method="isotonic")
            self._calibrator.fit(Xc, yc)
        return self

    def _proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self._pipe, "predict_proba") and hasattr(self._pipe.named_steps["clf"], "predict_proba"):
            return self._pipe.predict_proba(X)[:, 1]
        if hasattr(self._pipe.named_steps["clf"], "decision_function"):
            from sklearn.metrics import log_loss
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
