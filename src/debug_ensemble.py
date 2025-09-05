# src/debug_ensemble.py
from typing import Tuple
import pandas as pd
from .ensemble import Ensemble
from .execution import Execution

def debug_ensemble_trade(ens: Ensemble, exec: Execution, X_sample: pd.DataFrame) -> Tuple[float, str, object]:
    """
    Print base model probabilities, calibrated probabilities, ensemble prob,
    and simulate a single trade decision.
    """
    print("=== Ensemble Debug ===")
    
    # Base model raw probs
    for name, model in ens.members.items():
        raw_p = model._proba_raw(X_sample)
        print(f"[{name}] raw prob: {raw_p[0]:.3f}")
    
    # Calibrated probs (from Ensemble)
    print("\n[Calibrated probabilities per model]:")
    for name, model in ens.members.items():
        p = model.predict_proba(X_sample).iloc[0]
        if name in ens._calibrators:
            p_calib = ens._calibrators[name].transform([p])[0]
        else:
            p_calib = p
        print(f"[{name}] calibrated: {p_calib:.3f}")
    
    # Ensemble probability
    ensemble_prob = ens.predict_proba(X_sample).iloc[0]
    print(f"\n[Ensemble calibrated probability]: {ensemble_prob:.3f}")
    
    # Trade decision based on thresholds
    if ensemble_prob >= exec.prob_threshold_long:
        decision = "Going long"
    elif ensemble_prob <= exec.prob_threshold_short:
        decision = "Going short"
    else:
        decision = "Neutral zone, no trade"
    print(f"[Trade decision]: {decision}")
    
    # Optional: simulate actual order
    order_result = exec.market_order("EURUSD", X=X_sample)
    print(f"[Market order result]: {order_result}\n")
    
    return ensemble_prob, decision, order_result
