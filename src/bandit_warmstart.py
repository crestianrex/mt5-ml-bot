# src/bandit_warmstart.py
from __future__ import annotations
import json
import os
import glob
from typing import Any, Dict
from loguru import logger
import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Failed to load JSON from {path}: {e}")
        return {}


def _save_json(obj: Dict[str, Any], path: str) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
        logger.info(f"Saved merged bandit state to {path}")
    except Exception:
        logger.exception(f"Failed to save JSON to {path}")


def find_latest_backtest_state(results_dir: str = "results", pattern: str = "ts_risk_controller_state_backtest_*.json") -> str | None:
    """Find newest matching backtest state file in results_dir."""
    search = os.path.join(results_dir, pattern)
    matches = glob.glob(search)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _merge_numeric_lists(live: list, back: list, weight: float) -> list:
    # Add elementwise: live + weight * back; if sizes differ, pad with zeros
    n = max(len(live or []), len(back or []))
    out = []
    for i in range(n):
        lv = float(live[i]) if i < len(live or []) else 0.0
        bv = float(back[i]) if i < len(back or []) else 0.0
        out.append(lv + weight * bv)
    return out


def _merge_matrix(live: list, back: list, weight: float) -> list:
    # handle matrices stored as nested lists (A) or vectors (b)
    try:
        la = np.array(live) if live else np.zeros_like(back)
    except Exception:
        la = np.array(live or [])
    try:
        ba = np.array(back) if back else np.zeros_like(la)
    except Exception:
        ba = np.array(back or [])

    # If shapes differ, try to broadcast or create bigger array
    if la.size == 0:
        merged = (weight * ba).tolist()
    elif ba.size == 0:
        merged = la.tolist()
    else:
        # try to cast to same shape by padding
        try:
            # pad smaller to larger shape
            if la.shape != ba.shape:
                # create new array with shape = max dims
                new_shape = tuple(max(a, b) for a, b in zip(la.shape, ba.shape))
                new_la = np.zeros(new_shape)
                new_ba = np.zeros(new_shape)
                # copy contents
                new_la[tuple(slice(0, s) for s in la.shape)] = la
                new_ba[tuple(slice(0, s) for s in ba.shape)] = ba
                la, ba = new_la, new_ba
            merged = (la + weight * ba).tolist()
        except Exception:
            # fallback: elementwise flatten add
            la_flat = la.flatten()
            ba_flat = ba.flatten()
            merged = _merge_numeric_lists(list(la_flat), list(ba_flat), weight)
    return merged


def merge_warmstart(backtest_state_path: str | None, live_state_path: str, warmstart_weight: float = 1.0) -> None:
    """
    Merge a backtest bandit JSON into the live state file.
    - backtest_state_path: path to the backtest JSON (if None, try to auto-find)
    - live_state_path: path to the live TS JSON (cfg.thompson_sampling.state_file)
    - warmstart_weight: multiplier applied to backtest counts/rewards before adding to live.
    """
    if not backtest_state_path:
        logger.info("No backtest_state_path provided to merge_warmstart(); skipping.")
        return

    back = _load_json(backtest_state_path)
    if not back:
        logger.warning(f"No backtest state found at {backtest_state_path}; skipping warmstart merge.")
        return

    live = _load_json(live_state_path) if os.path.exists(live_state_path) else {}

    out = {}
    # Work on symbol_states level if exists, otherwise assume structure matches
    back_sym = back.get("symbol_states", back) if isinstance(back, dict) else {}
    live_sym = live.get("symbol_states", live) if isinstance(live, dict) else {}

    merged_sym_states = {}
    for symbol, bstate in (back_sym or {}).items():
        lstate = live_sym.get(symbol, {})
        merged = {}
        # Merge known bandit blocks
        for bandit_key in ["atr_bandit", "min_prob_bandit", "contextual_bandit"]:
            b_band = bstate.get(bandit_key, {})
            l_band = lstate.get(bandit_key, {})
            if not b_band and not l_band:
                continue
            merged_band = {}
            # numeric lists to merge
            for key in ["counts", "sum_rewards", "sum_squared_rewards"]:
                merged_band[key] = _merge_numeric_lists(l_band.get(key, []), b_band.get(key, []), warmstart_weight)
            # context matrices
            if "A" in b_band or "A" in l_band:
                merged_band["A"] = _merge_matrix(l_band.get("A", []), b_band.get("A", []), warmstart_weight)
            if "b" in b_band or "b" in l_band:
                merged_band["b"] = _merge_matrix(l_band.get("b", []), b_band.get("b", []), warmstart_weight)
            # copy over meta fields (num_arms, prior_mean, etc) - prefer live then backtest then defaults
            for meta in ["num_arms", "prior_mean", "prior_var", "min_var", "dim", "lambda_prior", "noise_var"]:
                if meta in l_band:
                    merged_band[meta] = l_band[meta]
                elif meta in b_band:
                    merged_band[meta] = b_band[meta]
            merged[bandit_key] = merged_band

        # Merge other top-level fields (peak_equity, current_equity, consecutive_losses, recent_returns, last_atr)
        for fld in ["peak_equity", "current_equity", "consecutive_losses", "last_atr"]:
            if fld in lstate:
                merged[fld] = lstate[fld]
            elif fld in bstate:
                merged[fld] = bstate[fld]
        # recent_returns: pick live's if present, else backtest's
        merged["recent_returns"] = lstate.get("recent_returns", bstate.get("recent_returns", []))

        merged_sym_states[symbol] = merged

    # If live had other symbols not in backtest, keep them
    for symbol, lstate in (live_sym or {}).items():
        if symbol not in merged_sym_states:
            merged_sym_states[symbol] = lstate

    out["symbol_states"] = merged_sym_states

    # persist merged object to live state file path
    _save_json(out, live_state_path)
