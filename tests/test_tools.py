from unittest.mock import patch

import pandas as pd
import pytest

from app.orchestration.tools import (
    ToolSafetyError,
    check_restricted_symbols,
    check_wash_sale_patterns,
    clean_data,
    fetch_market_data,
    run_analysis,
)


def _make_ohlcv(n: int = 20) -> list[dict]:
    rows = []
    price = 100.0
    for i in range(n):
        rows.append(
            {
                "time": f"2022-01-{i + 1:02d}",
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
                "adj_close": price + 0.5,
                "volume": 1000 + i * 10,
            }
        )
        price += 0.5
    return rows


# ── clean_data ────────────────────────────────────────────────────────────────


def test_clean_data_returns_dataframe():
    raw = {"ohlcv": _make_ohlcv(), "symbol": "AAPL"}
    result = clean_data(raw)
    assert isinstance(result["dataframe"], pd.DataFrame)
    assert not result["dataframe"].empty


def test_clean_data_empty_ohlcv():
    result = clean_data({"ohlcv": []})
    assert result["dataframe"].empty


def test_clean_data_time_parsed():
    raw = {"ohlcv": _make_ohlcv(5)}
    result = clean_data(raw)
    assert pd.api.types.is_datetime64_any_dtype(result["dataframe"]["time"])


# ── run_analysis ──────────────────────────────────────────────────────────────


def test_run_analysis_empty_dataframe():
    result = run_analysis({"dataframe": pd.DataFrame()})
    assert result["analysis"] == "No data"


def test_run_analysis_no_dataframe_key():
    result = run_analysis({})
    assert result["analysis"] == "No data"


def test_run_analysis_returns_rsi_and_signals():
    raw = {"ohlcv": _make_ohlcv(30)}
    cleaned = clean_data(raw)
    result = run_analysis(cleaned)
    assert "analysis" in result
    assert "signals" in result
    assert "risk_metrics" in result
    assert "rsi_last" in result["analysis"]
    assert "max_drawdown" in result["risk_metrics"]


def test_run_analysis_risk_metrics_bounds():
    raw = {"ohlcv": _make_ohlcv(30)}
    cleaned = clean_data(raw)
    result = run_analysis(cleaned)
    assert result["risk_metrics"]["max_drawdown"] <= 0.0
    assert result["risk_metrics"]["exposure"] == 1.0


# ── check_restricted_symbols ──────────────────────────────────────────────────


def test_check_restricted_symbols_pass():
    result = check_restricted_symbols(["AAPL", "GOOG", "MSFT"])
    assert result["status"] == "pass"
    assert result["violations"] == []


def test_check_restricted_symbols_empty():
    result = check_restricted_symbols([])
    assert result["status"] == "pass"


# ── check_wash_sale_patterns ──────────────────────────────────────────────────


def test_check_wash_sale_empty_trades():
    result = check_wash_sale_patterns([])
    assert result["status"] == "pass"


def test_check_wash_sale_non_empty_trades():
    result = check_wash_sale_patterns([{"symbol": "AAPL", "qty": 100}])
    assert result["status"] == "fail"


# ── fetch_market_data ─────────────────────────────────────────────────────────


def test_fetch_market_data_network_error():
    with patch("app.orchestration.tools.yf.download") as mock_dl:
        mock_dl.side_effect = Exception("connection refused")
        with pytest.raises(ToolSafetyError, match="yfinance failed"):
            fetch_market_data({"symbol": "AAPL", "start": "2022-01-01", "end": "2022-12-31"})


def test_fetch_market_data_empty_result():
    with patch("app.orchestration.tools.yf.download") as mock_dl:
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ToolSafetyError, match="No market data"):
            fetch_market_data({"symbol": "INVALID", "start": "2022-01-01", "end": "2022-12-31"})


def test_fetch_market_data_shape():
    sample = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2022-01-03", "2022-01-04"]),
    )
    sample.index.name = "Date"
    with patch("app.orchestration.tools.yf.download") as mock_dl:
        mock_dl.return_value = sample
        result = fetch_market_data({"symbol": "AAPL", "start": "2022-01-03", "end": "2022-01-04"})

    assert result["symbol"] == "AAPL"
    assert len(result["ohlcv"]) == 2
    assert "close" in result["ohlcv"][0]
    assert "open" in result["ohlcv"][0]
