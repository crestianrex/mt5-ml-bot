### src/ensemble.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
from .strategy_ml import MLStrategy
from sklearn.isotonic import IsotonicRegression
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

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
        self.ensemble_cv_auc_ = 0.50 # Initialize ensemble AUC

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
        logger.info("Fitting ensemble members with proper cross-validation...")
        """
        Fit all member models and the stacking meta-model using time-series cross-validation
        to prevent data leakage.
        """
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 300)))
        
        out_of_fold_predictions = []
        out_of_fold_true_values = []
        member_cv_aucs = {name: [] for name in self.members.keys()}
        meta_aucs = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            logger.debug(f"Fold {fold+1}/{tscv.n_splits}: Train size={len(X_tr)}, Val size={len(X_val)}")

            # Fit base models on the training part of the fold
            fold_base_model_preds = {}
            for name, model in self.members.items():
                model.fit(X_tr, y_tr)
                # Predict on the validation part of the fold
                p_val = model.predict_proba(X_val)
                fold_base_model_preds[name] = p_val
                
                # Store individual model AUC for this fold
                if hasattr(model, 'cv_auc_') and model.cv_auc_ is not None:
                     member_cv_aucs[name].append(model.cv_auc_)

            # Create the meta-model's training data for this fold
            P_val = pd.concat(fold_base_model_preds.values(), axis=1)
            P_val.columns = self.members.keys()

            # Store the out-of-fold predictions and true values
            out_of_fold_predictions.append(P_val)
            out_of_fold_true_values.append(y_val)

        # --- After all folds, create the full out-of-fold dataset ---
        P_oof = pd.concat(out_of_fold_predictions)
        y_oof = pd.concat(out_of_fold_true_values)

        # --- Fit the final meta-model on the clean out-of-fold predictions ---
        if self.method == "stacking":
            from sklearn.linear_model import LogisticRegression
            self._stacker = LogisticRegression(C=self.meta.get("C", 1.0), max_iter=200)
            self._stacker.fit(P_oof.values, y_oof.values)
            logger.info("Final stacking meta-model fitted on out-of-fold predictions.")

            # --- Calculate a more realistic ensemble CV AUC ---
            # We can do this by fitting a temp stacker on the first N-1 folds' OOF preds
            # and scoring on the last fold. This is a simplification but much better.
            for fold in range(1, tscv.n_splits):
                train_preds = pd.concat(out_of_fold_predictions[:fold])
                train_true = pd.concat(out_of_fold_true_values[:fold])
                val_preds = out_of_fold_predictions[fold]
                val_true = out_of_fold_true_values[fold]

                temp_stacker = LogisticRegression(C=self.meta.get("C", 1.0), max_iter=200)
                temp_stacker.fit(train_preds.values, train_true.values)
                p_meta_val = temp_stacker.predict_proba(val_preds.values)[:, 1]
                meta_aucs.append(roc_auc_score(y_val, p_meta_val))

            self.ensemble_cv_auc_ = float(np.mean(meta_aucs)) if meta_aucs else 0.5
            logger.info(f"Ensemble CV AUC (stacking, corrected): {self.ensemble_cv_auc_:.4f}")

            # --- Fit the final base models on ALL data for future predictions ---
            logger.info("Refitting base models on all data for future predict() calls...")
            for name, model in self.members.items():
                model.fit(X, y)
            
            # Optional isotonic calibration for the final stacker
            p_meta_full = self._stacker.predict_proba(P_oof.values)[:, 1]
            self._meta_calibrator = IsotonicRegression(out_of_bounds='clip')
            self._meta_calibrator.fit(p_meta_full, y_oof.values)
            logger.info("Meta-model isotonic calibration complete.")

        else: # For non-stacking methods
            member_aucs = [np.mean(aucs) for aucs in member_cv_aucs.values() if aucs]
            self.ensemble_cv_auc_ = float(np.mean(member_aucs)) if member_aucs else 0.5
            logger.info(f"Ensemble CV AUC (average members): {self.ensemble_cv_auc_:.4f}")

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

    
