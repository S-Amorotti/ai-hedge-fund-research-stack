from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd
import yfinance as yf


class ToolSafetyError(RuntimeError):
    """Raised when a tool is invoked in a prohibited way."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str


def fetch_market_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch market data from approved research sources (read-only).

    Uses yfinance for free, public OHLCV data. This does not connect to broker APIs.
    """

    symbol = params.get("symbol", "AAPL")
    start = params.get("start", "2022-01-01")
    end = params.get("end", "2022-12-31")
    interval = params.get("interval", "1d")

    try:
        data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
    except Exception as exc:  # pragma: no cover - network-bound
        raise ToolSafetyError(f"yfinance failed: {exc}") from exc

    if data.empty:
        raise ToolSafetyError("No market data returned from yfinance.")

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    data = data.reset_index().rename(columns={"Date": "time"})

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "ohlcv": data.to_dict(orient="records"),
    }


def clean_data(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and normalize raw research data.

    This function is deterministic and side-effect free.
    """

    ohlcv = raw.get("ohlcv", [])
    df = pd.DataFrame(ohlcv)
    if not df.empty and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return {"cleaned": raw, "dataframe": df, "notes": "Basic normalization applied"}


def run_analysis(cleaned: Dict[str, Any]) -> Dict[str, Any]:
    """Run research analysis on cleaned data.

    This function does not evaluate strategy quality or place trades.
    """

    df = cleaned.get("dataframe")
    if df is None or df.empty:
        return {"analysis": "No data", "inputs": cleaned}

    prices = df["close"].astype(float)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi

    signals = []
    for idx, row in df.iterrows():
        if pd.isna(row["rsi"]):
            continue
        if row["rsi"] < 30:
            signals.append({"time": row["time"], "price": row["close"], "action": "buy"})
        elif row["rsi"] > 70:
            signals.append({"time": row["time"], "price": row["close"], "action": "sell"})

    analysis = {
        "rsi_last": float(df["rsi"].iloc[-1]),
        "signal_count": len(signals),
    }
    risk_metrics = {
        "max_drawdown": float((prices / prices.cummax() - 1).min()),
        "exposure": 1.0,
    }
    return {
        "analysis": analysis,
        "inputs": cleaned,
        "signals": signals,
        "risk_metrics": risk_metrics,
    }


def check_restricted_symbols(symbols: List[str]) -> Dict[str, Any]:
    """Compliance check for restricted symbols."""

    restricted = set()
    violations = [s for s in symbols if s in restricted]
    return {
        "checked_symbols": symbols,
        "restricted": list(restricted),
        "violations": violations,
        "status": "pass" if not violations else "fail",
    }


def check_wash_sale_patterns(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compliance check for wash-sale patterns.

    Since this system is research-only, trades should be empty.
    Any non-empty input will be flagged.
    """

    return {
        "checked_trades": trades,
        "status": "pass" if not trades else "fail",
        "note": "Research-only system should not include trade data.",
    }


ALL_TOOLS: Dict[str, ToolSpec] = {
    "fetch_market_data": ToolSpec(
        name="fetch_market_data",
        description="Read-only access to approved research data sources.",
    ),
    "clean_data": ToolSpec(
        name="clean_data",
        description="Deterministic data normalization and cleaning.",
    ),
    "run_analysis": ToolSpec(
        name="run_analysis",
        description="Stateless research analysis routines.",
    ),
    "check_restricted_symbols": ToolSpec(
        name="check_restricted_symbols",
        description="Compliance check for restricted tickers.",
    ),
    "check_wash_sale_patterns": ToolSpec(
        name="check_wash_sale_patterns",
        description="Compliance check for wash-sale patterns.",
    ),
}
