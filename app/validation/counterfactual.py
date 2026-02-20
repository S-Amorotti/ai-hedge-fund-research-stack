from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass(frozen=True)
class CounterfactualConfig:
    scenarios: int = 50
    price_noise_std: float = 0.01
    earnings_shift_days: int = 3


def _shift_earnings_dates(dates: np.ndarray, shift: int) -> np.ndarray:
    return dates + shift


def generate_counterfactuals(
    dataset: Dict[str, Any],
    config: CounterfactualConfig = CounterfactualConfig(),
    seed: int = 7,
) -> List[Dict[str, Any]]:
    """Generate counterfactual datasets with deterministic perturbations."""

    rng = np.random.default_rng(seed)
    prices = np.asarray(dataset.get("prices", []), dtype=float)
    earnings = np.asarray(dataset.get("earnings_dates", []), dtype=int)
    sentiment = np.asarray(dataset.get("sentiment", []), dtype=float)

    counterfactuals: List[Dict[str, Any]] = []
    for i in range(config.scenarios):
        noise = rng.normal(0.0, config.price_noise_std, size=prices.shape)
        price_cf = prices + noise
        shift = rng.integers(-config.earnings_shift_days, config.earnings_shift_days + 1)
        earnings_cf = _shift_earnings_dates(earnings, int(shift))
        sentiment_cf = -sentiment

        counterfactuals.append(
            {
                "prices": price_cf,
                "earnings_dates": earnings_cf,
                "sentiment": sentiment_cf,
                "scenario_id": i,
            }
        )

    return counterfactuals
