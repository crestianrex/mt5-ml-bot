# src/ensemble.py
from __future__ import annotations
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from .strategy_ml import MLStrategy
from sklearn.isotonic import IsotonicRegression
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict


def custom_pnl(
    y_true: pd.Series,
    y_pred: pd.Series,
    prices: pd.Series,
    spread_pips: float = 2.0,
    slippage_pips: float = 0.5,
    commission_per_trade: float = 0.0,
    lot_size: float = 1.0,
    pip_value: float = 0.0001,
) -> float:
    """
    Simulate forex PnL given predictions, accounting for spread, commission, slippage.
    y_true: true labels (0/1)
    y_pred: binary predictions (0/1) or probabilities (>=0.5 => long)
    prices: Series of close prices aligned with y_true / y_pred (same index)
    """
    if prices is None or len(prices) != len(y_pred):
        raise ValueError(f"Prices and predictions must be same length: {len(prices)} vs {len(y_pred)}")

    pnl = []
    spread = spread_pips * pip_value
    slippage = slippage_pips * pip_value

    # iterate from i=1 so we have entry from previous bar
    for i in range(1, len(prices)):
        signal = 1 if y_pred.iloc[i] >= 0.5 else -1
        entry_price = prices.iloc[i - 1]
        exit_price = prices.iloc[i]

        # adjust for slippage
        if signal == 1:  # long
            entry_price_adj = entry_price + slippage
            exit_price_adj = exit_price - slippage
        else:  # short
            entry_price_adj = entry_price - slippage
            exit_price_adj = exit_price + slippage

        price_diff = exit_price_adj - entry_price_adj
        pip_diff = price_diff / pip_value

        ret = signal * pip_diff
        # subtract spread
        ret -= spread_pips
        # subtract commission (scaled by lot size; assume commission_per_trade is total cost)
        # normalize commission: if lot_size small, scale accordingly
        if lot_size > 0:
            ret -= commission_per_trade * (lot_size / max(lot_size, 1.0))
        pnl.append(ret * lot_size)

    total = float(np.sum(pnl))
    logger.debug(f"custom_pnl: total={total:.6f}, trades={len(pnl)}")
    return total

def calculate_sharpe_ratio(
    y_true: pd.Series,
    y_pred: pd.Series,
    prices: pd.Series,
    cfg: "Cfg",
    **trading_costs
) -> float:
    """Calculates annualized Sharpe ratio for a given set of predictions."""
    if prices is None or len(prices) != len(y_pred):
        raise ValueError(f"Prices and predictions must be same length: {len(prices)} vs {len(y_pred)}")

    pnl = []
    spread = trading_costs.get("spread_pips", 2.0) * trading_costs.get("pip_value", 0.0001)

    # Simulate PnL per bar, assuming a position is held for one bar
    for i in range(1, len(prices)):
        if y_pred.iloc[i-1] == 1: # Long signal
            pnl.append(prices.iloc[i] - prices.iloc[i-1] - spread)
        elif y_pred.iloc[i-1] == 0: # Short signal
            pnl.append(prices.iloc[i-1] - prices.iloc[i] - spread)
        else: # No signal
            pnl.append(0)

    returns = pd.Series(pnl)
    if returns.std() == 0 or returns.empty:
        return 0.0

    # Annualize Sharpe Ratio
    timeframe_minutes = cfg.timeframe_minutes()
    if timeframe_minutes is None:
        annualization_factor = 1.0 # Cannot determine timeframe, return raw Sharpe
    else:
        bars_per_year = 252 * (24 * 60 / timeframe_minutes) # 252 trading days
        annualization_factor = np.sqrt(bars_per_year)

    sharpe = (returns.mean() / returns.std()) * annualization_factor
    logger.debug(f"sharpe_ratio: calculated={sharpe:.4f}, returns={len(returns)}")
    return sharpe if np.isfinite(sharpe) else 0.0


class DynamicWeightedEnsemble:
    def __init__(self, base_models: Dict[str, MLStrategy], decay: float = 0.9, min_weight: float = 0.05):
        self.base_models = base_models
        self.decay = float(decay)
        self.min_weight = float(min_weight)
        self.model_scores: Dict[str, float] = {name: 0.5 for name in base_models}
        self.weights: Dict[str, float] = {name: 1.0 / len(base_models) for name in base_models}

    def update_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        new_scores: Dict[str, float] = {}
        for name, model in self.base_models.items():
            try:
                preds = model.predict_proba(X_val)
                if hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[1] >= 2:
                    p1 = preds[:, 1]
                else:
                    # fallback: if only single-dimension, treat as probability of up
                    p1 = np.array(preds).flatten()
                auc = roc_auc_score(y_val, p1)
            except Exception as e:
                logger.warning(f"DynamicWeightedEnsemble.update_weights: model {name} failed on validation: {e}")
                auc = 0.5
            # update with exponential decay
            previous = self.model_scores.get(name, 0.5)
            updated = self.decay * previous + (1.0 - self.decay) * auc
            self.model_scores[name] = updated
            new_scores[name] = updated

        # Normalize and apply min_weight flooring
        score_arr = np.array(list(new_scores.values()), dtype=float)
        # If sum very small, avoid division by zero
        sum_scores = score_arr.sum()
        if sum_scores <= 0:
            # fallback: equal weights
            logger.warning("DynamicWeightedEnsemble: sum of scores non-positive, falling back to equal weights.")
            self.weights = {name: 1.0 / len(new_scores) for name in new_scores}
            return self.weights

        # clip and renormalize
        weights = []
        for val in score_arr:
            w = max(val, self.min_weight)
            weights.append(w)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        self.weights = dict(zip(new_scores.keys(), weights))
        logger.debug(f"Updated dynamic weights: {self.weights}")
        return self.weights


class Ensemble:
    def __init__(self, cfg, model_params: Optional[Dict[str, Dict]] = None):
        """
        cfg: configuration object with
          - cfg.models: list of dicts, each with "name" and optional "params"
          - cfg.ensemble: method, weights, etc.
          - cfg.cv_samples_per_split, etc.
        model_params: optional override from tuning; keys matching model names.
        """
        self.cfg = cfg
        self.members: Dict[str, MLStrategy] = {}
        self.failed_members: set[str] = set()
        use_gpu = getattr(cfg, "use_gpu", False)
        self.cv_samples = getattr(cfg, "cv_samples_per_split", None) or 300

        for m in cfg.models:
            name = m.get("name")
            if not name:
                continue
            params = m.get("params", {}).copy()
            if model_params and name in model_params:
                params.update(model_params[name])

            # GPU device hint
            if use_gpu and name.lower() in ("lgbm", "xgb"):
                params["device"] = params.get("device", "gpu")

            try:
                self.members[name] = MLStrategy(model=name, calibrate=True, cv_samples_per_split=self.cv_samples, **params)
            except Exception as e:
                logger.error(f"Ensemble.__init__: failed to init member {name}: {e}")
                # do not include in members
                self.failed_members.add(name)

        if not self.members:
            raise ValueError("Ensemble requires at least one valid member model")

        self.method: str = self.cfg.ensemble.get("method", "soft_vote")
        self.weights: Dict[str, float] = self.cfg.ensemble.get("weights", {k: 1.0 / len(self.members) for k in self.members})
        self.meta: Dict = self.cfg.ensemble.get("meta", {"type": "logit", "C": 1.0})
        self.flat_mode: bool = bool(self.cfg.ensemble.get("flat_mode", False))
        self.threshold_metric: str = self.cfg.ensemble.get("threshold_metric", "custom_pnl")
        self.threshold_grid: str | float | List[float] = self.cfg.ensemble.get("threshold_grid", "auto")
        self.trading_costs = getattr(self.cfg, "trading_costs", {}).get("defaults", {})

        self._stacker = None
        self._meta_calibrator: Optional[IsotonicRegression] = None

        self.ensemble_cv_auc_: float = 0.5
        self.member_cv_aucs_: Dict[str, float] = {}
        self.dynamic_ensemble = DynamicWeightedEnsemble(self.members)

        # placeholder for threshold (after optimization)
        self.best_threshold_: Optional[float] = None

    def save(self, path: str):
        """Saves the entire ensemble to a directory."""
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving ensemble to {path}")

        # Save each member
        for name, member in self.members.items():
            member_path = os.path.join(path, name)
            try:
                member.save_model(member_path)
            except Exception as e:
                logger.error(f"Failed to save member {name}: {e}")

        # Save ensemble metadata
        meta_path = os.path.join(path, "ensemble_meta.pkl")
        metadata = {
            "ensemble_cv_auc_": self.ensemble_cv_auc_,
            "member_cv_aucs_": self.member_cv_aucs_,
            "best_threshold_": self.best_threshold_,
            "dynamic_ensemble_scores": self.dynamic_ensemble.model_scores,
            "dynamic_ensemble_weights": self.dynamic_ensemble.weights,
            "failed_members": self.failed_members,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        # Save stacker if it exists
        if self._stacker:
            stacker_path = os.path.join(path, "stacker.pkl")
            with open(stacker_path, "wb") as f:
                pickle.dump(self._stacker, f)

        # Save meta-calibrator if it exists
        if self._meta_calibrator:
            calibrator_path = os.path.join(path, "meta_calibrator.pkl")
            with open(calibrator_path, "wb") as f:
                pickle.dump(self._meta_calibrator, f)

    @classmethod
    def load(cls, path: str, cfg, model_params: Optional[Dict[str, Dict]] = None) -> "Ensemble":
        """Loads an entire ensemble from a directory."""
        logger.info(f"Loading ensemble from {path}")
        
        # Create a new ensemble instance to populate
        ensemble = cls(cfg, model_params=model_params)

        # Load each member
        for name, member in ensemble.members.items():
            member_path = os.path.join(path, name)
            if os.path.isdir(member_path):
                try:
                    member.load_model(member_path)
                except Exception as e:
                    logger.error(f"Failed to load member {name}: {e}")
                    ensemble.failed_members.add(name)
            else:
                logger.warning(f"Directory for member {name} not found at {member_path}")
                ensemble.failed_members.add(name)
        
        # Remove failed members from the active list
        for name in list(ensemble.failed_members):
            if name in ensemble.members:
                del ensemble.members[name]

        # Load ensemble metadata
        meta_path = os.path.join(path, "ensemble_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            ensemble.ensemble_cv_auc_ = metadata.get("ensemble_cv_auc_", 0.5)
            ensemble.member_cv_aucs_ = metadata.get("member_cv_aucs_", {})
            ensemble.best_threshold_ = metadata.get("best_threshold_")
            if hasattr(ensemble, "dynamic_ensemble"):
                ensemble.dynamic_ensemble.model_scores = metadata.get("dynamic_ensemble_scores", {k: 0.5 for k in ensemble.members})
                ensemble.dynamic_ensemble.weights = metadata.get("dynamic_ensemble_weights", {k: 1.0/len(ensemble.members) for k in ensemble.members})

        # Load stacker if it exists
        stacker_path = os.path.join(path, "stacker.pkl")
        if os.path.exists(stacker_path):
            with open(stacker_path, "rb") as f:
                ensemble._stacker = pickle.load(f)

        # Load meta-calibrator if it exists
        calibrator_path = os.path.join(path, "meta_calibrator.pkl")
        if os.path.exists(calibrator_path):
            with open(calibrator_path, "rb") as f:
                ensemble._meta_calibrator = pickle.load(f)
        
        return ensemble

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> Ensemble:
        logger.info("Ensemble.fit: start")

        if X is None or y is None:
            raise ValueError("X and y must be provided to fit()")

        # clean data
        Xc = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        yc = y.reindex(Xc.index)
        prices_c = None
        if prices is not None:
            prices_c = prices.reindex(Xc.index)
            if len(prices_c) != len(Xc):
                logger.warning("Prices length mismatches feature/label data in Ensemble.fit; dropping mismatched indices")
                # align
                common_idx = Xc.index.intersection(prices_c.index)
                Xc = Xc.loc[common_idx]
                yc = yc.loc[common_idx]
                prices_c = prices_c.loc[common_idx]

        n_samples = len(Xc)
        min_samples = getattr(self.cfg, "min_samples_for_ensemble", 200)
        if n_samples < min_samples:
            logger.warning(f"Not enough data to fit ensemble: {n_samples} samples < {min_samples}. Skipping fit.")
            # still attempt to fit members individually
            for name, model in self.members.items():
                try:
                    model.fit(Xc, yc)
                    # record individual cv_aucs_ if available
                    self.member_cv_aucs_[name] = getattr(model, "cv_auc_", 0.5)
                except Exception as e:
                    logger.warning(f"Ensemble.fit: member {name} failed to fit small data: {e}")
                    self.failed_members.add(name)
            self.ensemble_cv_auc_ = np.mean(list(self.member_cv_aucs_.values())) if self.member_cv_aucs_ else 0.5
            return self

        # TimeSeriesSplit CV
        cv_split = min(5, max(2, n_samples // self.cv_samples))
        tscv = TimeSeriesSplit(n_splits=cv_split)
        oof_preds: List[pd.Series] = []
        oof_true: List[pd.Series] = []
        member_cv_raw: Dict[str, List[float]] = {name: [] for name in self.members}

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(Xc)):
            X_tr, X_val = Xc.iloc[tr_idx], Xc.iloc[val_idx]
            y_tr, y_val = yc.iloc[tr_idx], yc.iloc[val_idx]

            if len(X_tr) < 20 or len(X_val) < 20:
                logger.debug(f"Skip fold {fold_idx} due to too small split: {len(X_tr)}/{len(X_val)}")
                continue

            # Fit each member
            fold_base_preds: Dict[str, pd.Series] = {}
            for name, model in self.members.items():
                try:
                    if not model.fit(X_tr, y_tr):
                        logger.warning(f"Ensemble.fit: member {name} skipped fitting in fold {fold_idx} due to insufficient data.")
                        continue
                    proba = model.predict_proba(X_val)
                    # ensure proba is shaped correctly
                    if hasattr(proba, "ndim") and proba.ndim == 2:
                        p1 = proba[:, 1]
                    else:
                        p1 = np.array(proba).flatten()
                    fold_base_preds[name] = pd.Series(p1, index=y_val.index, name=name)
                    # record CV score
                    auc = roc_auc_score(y_val, p1)
                    member_cv_raw[name].append(auc)
                except Exception as e:
                    logger.warning(f"Ensemble.fit: member {name} failed in fold {fold_idx}: {e}")
                    self.failed_members.add(name)

            if not fold_base_preds:
                logger.warning(f"Ensemble.fit: no valid member predictions in fold {fold_idx}")
                continue

            # Aggregate OOF
            P_val_df = pd.concat(fold_base_preds.values(), axis=1)
            # Ensure P_val_df columns correspond to member names
            P_val_df.columns = list(fold_base_preds.keys())

            oof_preds.append(P_val_df)
            oof_true.append(y_val)

            # Update dynamic weights
            self.dynamic_ensemble.update_weights(X_val, y_val)

        if not oof_preds:
            # no valid folds
            logger.warning("Ensemble.fit: no valid CV folds; skipping threshold optimization.")
            self.ensemble_cv_auc_ = np.mean([np.mean(v) for v in member_cv_raw.values() if v]) if any(member_cv_raw.values()) else 0.5
        else:
            P_oof = pd.concat(oof_preds)
            y_oof = pd.concat(oof_true)
            # Log member CV AUC
            for name, auc_list in member_cv_raw.items():
                if auc_list:
                    self.member_cv_aucs_[name] = float(np.mean(auc_list))
                else:
                    self.member_cv_aucs_[name] = 0.5
                logger.info(f"[{name}] CV AUC: {self.member_cv_aucs_[name]:.4f}")

            if self.method == "stacking":
                from sklearn.linear_model import LogisticRegression

                try:
                    self._stacker = LogisticRegression(C=self.meta.get("C", 1.0), max_iter=200)
                    self._stacker.fit(P_oof.values, y_oof.values)
                    proba_stacked = self._stacker.predict_proba(P_oof.values)[:, 1]
                    self.ensemble_cv_auc_ = roc_auc_score(y_oof, proba_stacked)
                except Exception as e:
                    logger.error(f"Ensemble.fit: stacking failed: {e}")
                    # fallback: average of member CVs
                    self.ensemble_cv_auc_ = float(np.mean(list(self.member_cv_aucs_.values()))
                                                 if self.member_cv_aucs_ else 0.5)
                    self._stacker = None
            else:
                # using soft vote or other methods
                self.ensemble_cv_auc_ = float(np.mean(list(self.member_cv_aucs_.values())))

            logger.info(f"Ensemble CV AUC: {self.ensemble_cv_auc_:.4f}")

            # threshold optimization if prices aligned
            if prices_c is not None:
                try:
                    # use last len(y_oof) prices
                    price_segment = prices_c.iloc[-len(y_oof):]
                    # prepare average base model proba across folds
                    # compute mean predictions across folds per sample
                    # simplest: use P_oof.mean(axis=1)
                    mean_proba = P_oof.mean(axis=1)
                    self._optimize_threshold(y_oof, mean_proba, price_segment)
                except Exception as e:
                    logger.warning(f"Ensemble.fit: threshold optimization failed: {e}")

        # After CV, refit members on all data
        for name, model in self.members.items():
            try:
                model.fit(Xc, yc)
            except Exception as e:
                logger.warning(f"Ensemble.fit: member {name} failed full data fit: {e}")
                self.failed_members.add(name)

        return self

    def _optimize_threshold(self, y_true: pd.Series, y_pred_probs: pd.Series, prices: pd.Series) -> Optional[float]:
        if prices is None or len(y_true) != len(y_pred_probs) or len(prices) != len(y_pred_probs):
            logger.warning("Threshold optimization: input lengths mismatch; skipping optimization.")
            return None

        if self.threshold_grid == "auto":
            thresholds = np.linspace(0.3, 0.7, 21)
        elif isinstance(self.threshold_grid, (list, np.ndarray)):
            thresholds = np.array(self.threshold_grid, dtype=float)
        elif isinstance(self.threshold_grid, (float, int)):
            thresholds = np.array([float(self.threshold_grid)])
        else:
            thresholds = np.linspace(0.0, 1.0, 101)

        best_thr = 0.5
        best_score = -np.inf

        for thr in thresholds:
            preds = (y_pred_probs >= thr).astype(int)
            score: float
            if self.threshold_metric == "f1":
                score = f1_score(y_true, preds)
            elif self.threshold_metric == "precision":
                score = precision_score(y_true, preds)
            elif self.threshold_metric == "recall":
                score = recall_score(y_true, preds)
            elif self.threshold_metric == "custom_pnl":
                try:
                    score = custom_pnl(y_true, preds, prices, **self.trading_costs)
                except Exception as e:
                    logger.warning(f"Threshold evaluation custom_pnl failed at thr={thr}: {e}")
                    continue
            elif self.threshold_metric == "sharpe_ratio":
                try:
                    score = calculate_sharpe_ratio(y_true, preds, prices, self.cfg, **self.trading_costs)
                except Exception as e:
                    logger.warning(f"Threshold evaluation sharpe_ratio failed at thr={thr}: {e}")
                    continue
            else:
                score = f1_score(y_true, preds)

            if score > best_score:
                best_score = score
                best_thr = thr

        self.best_threshold_ = best_thr
        logger.info(f"Optimized threshold: {best_thr:.3f} (metric={self.threshold_metric}, best score={best_score:.4f})")
        return best_thr

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if X is None:
            raise ValueError("X must be provided to predict_proba()")

        # clean features
        Xc = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        if len(Xc) == 0:
            logger.warning("predict_proba: empty features after cleaning, returning default 0.5")
            return pd.Series(0.5, index=X.index, name="p_up")

        # Collect member probabilities
        member_probs: Dict[str, pd.Series] = {}
        for name, model in self.members.items():
            try:
                proba = model.predict_proba(Xc)
                if hasattr(proba, "ndim") and proba.ndim == 2:
                    p1 = proba[:, 1]
                else:
                    p1 = np.array(proba).flatten()
                member_probs[name] = pd.Series(p1, index=Xc.index, name=name)
            except Exception as e:
                logger.warning(f"Ensemble.predict_proba: member {name} failed proba: {e}")
                # fallback: constant 0.5
                member_probs[name] = pd.Series(0.5, index=Xc.index, name=name)

        P_df = pd.concat(member_probs.values(), axis=1)
        P_df.columns = list(member_probs.keys())

        if self.method == "soft_vote":
            w = np.array([self.dynamic_ensemble.weights.get(k, 1.0 / len(self.members)) for k in P_df.columns], dtype=float)
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            # weighted average of p_up
            p_final = (P_df.values * w).sum(axis=1)
        elif self.method == "stacking" and self._stacker is not None:
            # stacker expects 2-D array
            try:
                p_final = self._stacker.predict_proba(P_df.values)[:, 1]
            except Exception as e:
                logger.error(f"Ensemble.predict_proba: stacking predict failed: {e}")
                # fallback to soft vote
                p_final = P_df.mean(axis=1).values
        else:
            # fallback: simple average
            p_final = P_df.mean(axis=1).values

        return pd.Series(p_final, index=P_df.index, name="p_up")