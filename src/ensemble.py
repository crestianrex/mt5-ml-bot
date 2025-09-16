from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
from .strategy_ml import MLStrategy
from sklearn.isotonic import IsotonicRegression
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict

# ==============================
#   Forex-Aware PnL Function
# ==============================
def custom_pnl(y_true, y_pred, prices: pd.Series,
               spread_pips=2.0, slippage_pips=0.5, commission_per_trade=0.0,
               lot_size=1.0, pip_value=0.0001) -> float:
    """
    Simulate forex PnL given predictions, accounting for spread, commission, slippage.

    y_true: true labels (unused, but kept for consistency with sklearn scorer API)
    y_pred: binary predictions or probs (0/1 long-short signals)
    prices: pd.Series of close prices
    """
    if len(prices) != len(y_pred):
        raise ValueError("Prices and predictions length mismatch")

    pnl = []
    spread = spread_pips * pip_value
    slippage = slippage_pips * pip_value

    for i in range(1, len(prices)):
        signal = 1 if y_pred[i] >= 0.5 else -1  # long/short
        entry_price = prices.iloc[i - 1]
        exit_price = prices.iloc[i]

        # adjust for slippage
        if signal == 1:  # long
            entry_price += slippage
            exit_price -= slippage
        else:            # short
            entry_price -= slippage
            exit_price += slippage

        # profit in pips
        pip_diff = (exit_price - entry_price) / pip_value
        ret = pip_diff * signal

        # subtract spread and commission
        ret -= spread_pips
        ret -= commission_per_trade / (lot_size / 0.01)  # normalize commission

        pnl.append(ret * lot_size)

    return float(np.sum(pnl))


# ==============================
#   Dynamic Weighted Ensemble
# ==============================
class DynamicWeightedEnsemble:
    def __init__(self, base_models, decay=0.9, min_weight=0.05):
        """
        base_models: dict {name: model}
        decay: exponential decay factor for old scores
        min_weight: floor weight to prevent total exclusion
        """
        self.base_models = base_models
        self.decay = decay
        self.min_weight = min_weight
        self.model_scores = defaultdict(lambda: 0.5)  # init with neutral AUC
        self.weights = {name: 1.0 / len(base_models) for name in base_models}

    def update_weights(self, X_val, y_val):
        """Update model weights based on latest validation performance."""
        new_scores = {}
        for name, model in self.base_models.items():
            try:
                preds = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, preds)
            except Exception:
                auc = 0.5  # fallback
            self.model_scores[name] = (
                self.decay * self.model_scores[name] + (1 - self.decay) * auc
            )
            new_scores[name] = self.model_scores[name]

        # normalize weights
        scores = np.array(list(new_scores.values()))
        weights = scores / (scores.sum() + 1e-9)

        # apply min_weight
        weights = np.clip(weights, self.min_weight, None)
        weights = weights / weights.sum()

        self.weights = dict(zip(new_scores.keys(), weights))
        return self.weights

    def predict_proba(self, X):
        """Weighted soft voting using dynamic weights."""
        preds = np.zeros(len(X))
        for name, model in self.base_models.items():
            p = model.predict_proba(X)[:, 1]
            preds += self.weights.get(name, 1.0 / len(self.base_models)) * p
        return np.vstack([1 - preds, preds]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


# ==============================
#   Main Ensemble Class
# ==============================
class Ensemble:
    def __init__(self, cfg, model_params: dict | None = None):
        self.cfg = cfg
        self.members: Dict[str, MLStrategy] = {}
        use_gpu = cfg.use_gpu
        cv_samples_per_split = cfg.cv_samples_per_split

        for m in cfg.models:
            name = m["name"]
            params = m.get("params", {}).copy()
            if model_params and name in model_params:
                params.update(model_params[name])

            # --- GPU Acceleration ---
            if use_gpu and name in ["lgbm", "xgb"]:
                params['device'] = 'gpu'

            self.members[name] = MLStrategy(model=name, 
                                          calibrate=True, 
                                          cv_samples_per_split=cv_samples_per_split, 
                                          **params)

        # Ensemble configuration
        self.method = cfg.ensemble.get("method", "soft_vote")
        self.weights = cfg.ensemble.get("weights", {k: 1/len(self.members) for k in self.members})
        self.meta = cfg.ensemble.get("meta", {"type": "logit", "C": 1.0})
        self.flat_mode = cfg.ensemble.get("flat_mode", False)
        self.threshold_metric = cfg.ensemble.get("threshold_metric", "custom_pnl")
        self.threshold_grid = cfg.ensemble.get("threshold_grid", "auto")

        # Trading cost parameters from config.yaml
        self.trading_costs = cfg.trading_costs.get("defaults") if hasattr(cfg, 'trading_costs') else {}

        # Internal trackers
        self._stacker = None
        self._meta_calibrator: IsotonicRegression | None = None
        self.ensemble_cv_auc_ = 0.50
        self.member_cv_aucs_: Dict[str, float] = {}

        # Dynamic weighting engine
        self.dynamic_ensemble = DynamicWeightedEnsemble(self.members)

    def fit(self, X: pd.DataFrame, y: pd.Series, prices: pd.Series | None = None):
        logger.info("Fitting ensemble members with time-series CV...")
        self.member_cv_aucs_ = {}

        cv_samples_per_split = self.cfg.cv_samples_per_split
        n_splits = min(5, max(2, len(X) // cv_samples_per_split))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        out_of_fold_predictions = []
        out_of_fold_true_values = []
        member_cv_aucs_raw = {name: [] for name in self.members.keys()}
        meta_aucs = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            # fit members
            fold_base_preds = {}
            for name, model in self.members.items():
                model.fit(X_tr, y_tr)
                p_val = model.predict_proba(X_val)
                fold_base_preds[name] = p_val
                if hasattr(model, 'cv_auc_') and model.cv_auc_ is not None:
                    member_cv_aucs_raw[name].append(model.cv_auc_)

            # stacking features
            P_val = pd.concat(fold_base_preds.values(), axis=1)
            P_val.columns = self.members.keys()

            out_of_fold_predictions.append(P_val)
            out_of_fold_true_values.append(y_val)

            # update dynamic weights
            self.dynamic_ensemble.update_weights(X_val, y_val)

        # aggregate OOF
        P_oof = pd.concat(out_of_fold_predictions)
        y_oof = pd.concat(out_of_fold_true_values)

        # log average AUC per member
        for name, aucs in member_cv_aucs_raw.items():
            self.member_cv_aucs_[name] = float(np.mean(aucs)) if aucs else 0.5
            logger.info(f"[{name}] CV AUC: {self.member_cv_aucs_[name]:.4f}")

        # ensemble CV metric
        if self.method == "stacking":
            from sklearn.linear_model import LogisticRegression
            self._stacker = LogisticRegression(C=self.meta.get("C", 1.0), max_iter=200)
            self._stacker.fit(P_oof.values, y_oof.values)
            logger.info("Stacking meta-model fitted.")
            self.ensemble_cv_auc_ = roc_auc_score(y_oof, self._stacker.predict_proba(P_oof.values)[:, 1])
        else:
            self.ensemble_cv_auc_ = float(np.mean(list(self.member_cv_aucs_.values())))
            logger.info(f"Ensemble CV metric: {self.ensemble_cv_auc_:.4f}")

        # refit base models on all data
        for name, model in self.members.items():
            model.fit(X, y)

        # threshold optimization
        if prices is not None:
            self._optimize_threshold(y_oof, P_oof.mean(axis=1), prices.iloc[-len(y_oof):])

        return self

    def _optimize_threshold(self, y_true, y_pred_probs, prices: pd.Series):
        """Optimize classification threshold based on chosen metric."""
        if self.threshold_grid == "auto":
            thresholds = np.linspace(0.3, 0.7, 21)  # focus around 0.5
        else:
            thresholds = np.linspace(0.0, 1.0, 101)

        best_thr, best_score = 0.5, -np.inf
        for thr in thresholds:
            preds = (y_pred_probs >= thr).astype(int)

            if self.threshold_metric == "f1":
                score = f1_score(y_true, preds)
            elif self.threshold_metric == "precision":
                score = precision_score(y_true, preds)
            elif self.threshold_metric == "recall":
                score = recall_score(y_true, preds)
            elif self.threshold_metric == "custom_pnl":
                score = custom_pnl(
                    y_true, preds, prices,
                    **self.trading_costs
                )
            else:
                score = f1_score(y_true, preds)

            if score > best_score:
                best_thr, best_score = thr, score

        self.best_threshold_ = best_thr
        logger.info(f"Optimized threshold: {best_thr:.3f} (metric={self.threshold_metric}, score={best_score:.4f})")
        return best_thr

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return calibrated probability of 'up' for ensemble"""
        Pcols = [m.predict_proba(X).rename(n) for n, m in self.members.items()]
        P = pd.concat(Pcols, axis=1)

        if self.method == "soft_vote":
            w = np.array([self.dynamic_ensemble.weights.get(k, 1.0) for k in P.columns], dtype=float)
            w /= w.sum()
            p_final = (P.values * w).sum(axis=1)

        elif self.method == "stacking" and self._stacker is not None:
            p_final = self._stacker.predict_proba(P.values)[:, 1]

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