import numpy as np
import pandas as pd
import pytest

from app.validation.counterfactual import CounterfactualConfig, generate_counterfactuals
from app.validation.metrics import build_consistency_report, prediction_consistency
from app.validation.factfin_validator import FactFinConfig, FactFinValidator


# ── generate_counterfactuals ──────────────────────────────────────────────────


def test_generate_counterfactuals_default_count():
    result = generate_counterfactuals({})
    assert len(result) == CounterfactualConfig().scenarios


def test_generate_counterfactuals_with_prices():
    dataset = {
        "prices": np.array([100.0, 101.0, 102.0, 101.5, 103.0]),
        "earnings_dates": np.array([1, 3]),
        "sentiment": np.array([0.1, -0.2, 0.3, 0.0, -0.1]),
    }
    result = generate_counterfactuals(dataset)
    assert len(result) == 50
    for cf in result:
        assert "prices" in cf
        assert "earnings_dates" in cf
        assert "sentiment" in cf
        assert "scenario_id" in cf


def test_generate_counterfactuals_sentiment_inverted():
    dataset = {"sentiment": np.array([0.5, -0.3])}
    result = generate_counterfactuals(dataset)
    np.testing.assert_array_almost_equal(result[0]["sentiment"], np.array([-0.5, 0.3]))


def test_generate_counterfactuals_deterministic():
    dataset = {"prices": np.array([100.0, 101.0, 102.0])}
    r1 = generate_counterfactuals(dataset, seed=99)
    r2 = generate_counterfactuals(dataset, seed=99)
    np.testing.assert_array_equal(r1[0]["prices"], r2[0]["prices"])


def test_generate_counterfactuals_custom_config():
    config = CounterfactualConfig(scenarios=5)
    result = generate_counterfactuals({}, config=config)
    assert len(result) == 5


# ── prediction_consistency ────────────────────────────────────────────────────


def test_prediction_consistency_perfect_agreement():
    baseline = np.array([1.0, -1.0, 1.0])
    cf = [np.array([2.0, -0.5, 0.1])]
    assert prediction_consistency(baseline, cf) == pytest.approx(1.0)


def test_prediction_consistency_total_disagreement():
    baseline = np.array([1.0, -1.0, 1.0])
    cf = [np.array([-2.0, 0.5, -0.1])]
    assert prediction_consistency(baseline, cf) == pytest.approx(0.0)


def test_prediction_consistency_empty_baseline():
    assert prediction_consistency(np.array([]), [np.array([1.0])]) == 0.0


def test_prediction_consistency_empty_counterfactuals():
    assert prediction_consistency(np.array([1.0]), []) == 0.0


def test_prediction_consistency_size_mismatch():
    baseline = np.array([1.0, -1.0])
    cf = [np.array([1.0])]
    assert prediction_consistency(baseline, cf) == pytest.approx(0.0)


def test_prediction_consistency_multiple_scenarios():
    baseline = np.array([1.0, -1.0])
    cf = [
        np.array([1.0, -1.0]),
        np.array([-1.0, 1.0]),
    ]
    assert prediction_consistency(baseline, cf) == pytest.approx(0.5)


# ── build_consistency_report ──────────────────────────────────────────────────


def test_build_consistency_report_structure():
    report = build_consistency_report(np.array([1.0]), [np.array([1.0])])
    assert "prediction_consistency" in report
    assert 0.0 <= report["prediction_consistency"] <= 1.0


# ── FactFinValidator ──────────────────────────────────────────────────────────


def _sample_df(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    price = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": price - 0.5,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "adj_close": price,
            "volume": [1000] * n,
        },
        index=idx,
    )


def test_factfin_perturb_data_count():
    validator = FactFinValidator(FactFinConfig(scenarios=3))
    df = _sample_df()
    scenarios = validator.perturb_data(df)
    assert len(scenarios) == 3


def test_factfin_perturb_data_empty():
    validator = FactFinValidator()
    scenarios = validator.perturb_data(pd.DataFrame())
    assert scenarios == []


def test_factfin_perturb_data_prices_differ():
    validator = FactFinValidator(FactFinConfig(scenarios=2))
    df = _sample_df()
    scenarios = validator.perturb_data(df, seed=42)
    assert not scenarios[0]["close"].equals(df["close"])


def test_factfin_consistency_check_empty_df():
    validator = FactFinValidator()
    result = validator.consistency_check(lambda d: {}, pd.DataFrame())
    assert result["flagged"] is True
    assert result["prediction_consistency"] == 0.0


def test_factfin_walk_forward_empty_df():
    validator = FactFinValidator()
    sharpe_df, heatmap = validator.walk_forward_optimization(pd.DataFrame())
    assert sharpe_df.empty
    assert heatmap.empty


def test_factfin_walk_forward_produces_sharpe():
    validator = FactFinValidator()
    df = _sample_df(n=200)
    sharpe_df, heatmap = validator.walk_forward_optimization(df)
    assert not sharpe_df.empty
    assert "sharpe" in sharpe_df.columns
