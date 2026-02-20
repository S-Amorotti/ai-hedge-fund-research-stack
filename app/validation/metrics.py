from __future__ import annotations

from typing import Dict, Any, List
import numpy as np


def prediction_consistency(
    baseline: np.ndarray, counterfactual_predictions: List[np.ndarray]
) -> float:
    """Compute Prediction Consistency (PC) in [0, 1].

    PC is the average fraction of predictions that match the baseline sign
    across counterfactual scenarios.
    """

    if baseline.size == 0 or not counterfactual_predictions:
        return 0.0

    baseline_sign = np.sign(baseline)
    matches = []
    for cf in counterfactual_predictions:
        if cf.size != baseline.size:
            matches.append(0.0)
            continue
        cf_sign = np.sign(cf)
        matches.append(float(np.mean(cf_sign == baseline_sign)))

    return float(np.mean(matches))


def build_consistency_report(
    baseline: np.ndarray, counterfactual_predictions: List[np.ndarray]
) -> Dict[str, Any]:
    pc = prediction_consistency(baseline, counterfactual_predictions)
    return {"prediction_consistency": pc}
