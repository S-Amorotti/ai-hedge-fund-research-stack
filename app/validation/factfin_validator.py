from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactFinConfig:
    scenarios: int = 10
    price_noise_std: float = 0.01
    earnings_shift_days: int = 3
    consistency_threshold: float = 0.7


class FactFinValidator:
    """Validation utilities to reduce profit-mirage risks."""

    def __init__(self, config: FactFinConfig | None = None) -> None:
        self.config = config or FactFinConfig()

    def perturb_data(self, df: pd.DataFrame, seed: int = 7) -> List[pd.DataFrame]:
        """Generate counterfactual scenarios with Gaussian noise and earnings shifts."""

        rng = np.random.default_rng(seed)
        scenarios: List[pd.DataFrame] = []
        if df.empty:
            return scenarios

        base = df.copy()
        if "earnings_date" not in base.columns:
            base["earnings_date"] = pd.NaT

        for _ in range(self.config.scenarios):
            cf = base.copy()
            for col in ["open", "high", "low", "close", "adj_close"]:
                if col in cf.columns:
                    noise = rng.normal(0.0, self.config.price_noise_std, size=len(cf))
                    cf[col] = cf[col].astype(float) * (1.0 + noise)

            if "earnings_date" in cf.columns and cf["earnings_date"].notna().any():
                shift = rng.integers(
                    -self.config.earnings_shift_days,
                    self.config.earnings_shift_days + 1,
                )
                cf["earnings_date"] = cf["earnings_date"] + pd.to_timedelta(int(shift), unit="D")

            scenarios.append(cf)

        return scenarios

    def _extract_recommendation(self, result: Dict[str, Any]) -> np.ndarray:
        if "prediction" in result:
            return np.atleast_1d(result["prediction"]).astype(float)
        if "signal" in result:
            return np.atleast_1d(result["signal"]).astype(float)
        if "recommendation" in result:
            rec = result["recommendation"]
            mapping = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
            return np.atleast_1d(mapping.get(str(rec).lower(), 0.0)).astype(float)
        return np.array([], dtype=float)

    def consistency_check(
        self,
        executor: Callable[[pd.DataFrame], Dict[str, Any]],
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run executor on perturbed scenarios and compute Prediction Consistency."""

        scenarios = self.perturb_data(df)
        if not scenarios:
            return {"prediction_consistency": 0.0, "flagged": True, "reason": "empty data"}

        baseline = self._extract_recommendation(executor(df))
        if baseline.size == 0:
            return {
                "prediction_consistency": 0.0,
                "flagged": True,
                "reason": "missing baseline prediction",
            }

        matches = []
        for scenario in scenarios:
            prediction = self._extract_recommendation(executor(scenario))
            if prediction.size != baseline.size:
                matches.append(0.0)
                continue
            matches.append(float(np.mean(np.sign(prediction) == np.sign(baseline))))

        pc = float(np.mean(matches)) if matches else 0.0
        flagged = pc < self.config.consistency_threshold
        return {
            "prediction_consistency": pc,
            "flagged": flagged,
            "threshold": self.config.consistency_threshold,
        }

    def walk_forward_optimization(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Rolling window backtest: train on months 1-6, test on month 7, etc."""

        if df.empty or "close" not in df.columns:
            return pd.DataFrame(), pd.DataFrame()

        data = df.copy()
        data = data.sort_index()
        data["month"] = data.index.to_period("M")
        months = sorted(data["month"].unique())

        def default_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
            # Simple momentum: long if last train return positive, else flat.
            train_ret = train_df["close"].pct_change().mean()
            position = 1.0 if train_ret > 0 else 0.0
            return test_df["close"].pct_change().fillna(0.0) * position

        strategy_fn = strategy_fn or default_strategy

        sharpe_rows: List[Dict[str, Any]] = []
        for i in range(6, len(months)):
            train_months = months[i - 6 : i]
            test_month = months[i]
            train_df = data[data["month"].isin(train_months)]
            test_df = data[data["month"] == test_month]
            returns = strategy_fn(train_df, test_df)
            if returns.empty or returns.std() == 0:
                sharpe = 0.0
            else:
                sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
            sharpe_rows.append(
                {"train_window": f"{train_months[0]}-{train_months[-1]}", "test_month": str(test_month), "sharpe": sharpe}
            )

        if not sharpe_rows:
            return pd.DataFrame(), pd.DataFrame()

        sharpe_df = pd.DataFrame(sharpe_rows)
        heatmap = sharpe_df.pivot(index="train_window", columns="test_month", values="sharpe")
        return sharpe_df, heatmap
