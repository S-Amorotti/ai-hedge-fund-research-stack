from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from graphviz import Digraph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.monitoring.log_writer import read_logs  # noqa: E402
from app.orchestration.graph import run_graph  # noqa: E402

PAUSE_FLAG = os.getenv("PAUSE_FLAG", "app/monitoring/pause.flag")
APPROVAL_FLAG = os.getenv("APPROVAL_FLAG", "app/monitoring/approval.flag")

st.set_page_config(page_title="AI Hedge Fund Monitor", layout="wide")
st.title("AI Hedge Fund Monitoring")

try:
    st.autorefresh(interval=2000, key="live_refresh")
except Exception:
    pass


def _build_flow_graph(active_node: str | None) -> Digraph:
    graph = Digraph()
    nodes = [
        ("planner", "Planner"),
        ("executor", "Executor"),
        ("critic", "Critic"),
        ("risk", "Risk Manager"),
        ("approval", "Human Approval"),
    ]
    edges = [
        ("planner", "executor"),
        ("executor", "critic"),
        ("critic", "risk"),
        ("risk", "approval"),
    ]
    for node_id, label in nodes:
        color = "red" if node_id == active_node else "black"
        graph.node(node_id, label=label, color=color, fontcolor=color)
    for start, end in edges:
        graph.edge(start, end)
    return graph


def _render_market_chart(market_data: Dict[str, Any]) -> None:
    ohlcv = market_data.get("ohlcv", [])
    if not ohlcv:
        st.info("No market data available yet.")
        return

    df = pd.DataFrame(ohlcv)
    if df.empty or not {"time", "open", "high", "low", "close"}.issubset(df.columns):
        st.info("Market data missing required columns.")
        return

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            )
        ]
    )

    signals = market_data.get("signals", [])
    if signals:
        signal_df = pd.DataFrame(signals)
        if {"time", "price", "action"}.issubset(signal_df.columns):
            colors = signal_df["action"].map({"buy": "green", "sell": "red"}).fillna("blue")
            fig.add_trace(
                go.Scatter(
                    x=signal_df["time"],
                    y=signal_df["price"],
                    mode="markers",
                    marker=dict(color=colors, size=10, symbol="triangle-up"),
                    name="Signals",
                )
            )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time",
        yaxis_title="Price",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_gauge(confidence: float) -> None:
    gauge_color = "red" if confidence < 0.7 else "green"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [{"range": [0, 70], "color": "#f4cccc"}],
            },
            title={"text": "Confidence Score"},
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _write_approval(value: str) -> None:
    with open(APPROVAL_FLAG, "w", encoding="utf-8") as handle:
        handle.write(value)


logs: List[Dict[str, Any]] = read_logs()
latest = logs[-1] if logs else {}

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Control")
    strategy = st.text_input("Strategy Hypothesis", value="Hypothesis: earnings sentiment predicts short-term drift")
    date_start = st.date_input("Start Date", value=datetime(2022, 1, 1))
    date_end = st.date_input("End Date", value=datetime(2022, 12, 31))

    if st.button("Run Research"):
        try:
            run_graph(strategy)
        except RuntimeError as exc:
            st.error(str(exc))

    if st.button("Kill Switch"):
        with open(PAUSE_FLAG, "w", encoding="utf-8") as handle:
            handle.write("paused")

    if st.button("Resume"):
        if os.path.exists(PAUSE_FLAG):
            os.remove(PAUSE_FLAG)

    st.caption("Kill Switch is the Pause flag. Execution halts safely on detection.")

with col2:
    st.subheader("Flow Visualization")
    active_node = latest.get("active_node")
    st.graphviz_chart(_build_flow_graph(active_node))

    with st.expander("Thought Trace", expanded=False):
        messages = latest.get("messages", [])
        if messages:
            for msg in messages[-5:]:
                st.write(f"{msg.get('role', 'agent')}: {msg.get('content', '')}")
        else:
            st.write("No thought trace available.")

with col3:
    st.subheader("Market View")
    _render_market_chart(latest.get("market_data", {}))
    _render_confidence_gauge(float(latest.get("confidence", 0.0)))

awaiting_approval = bool(latest.get("awaiting_approval"))
if awaiting_approval:
    with st.dialog("Review & Approve"):
        st.write("The system is waiting for approval. Review the proposed code below.")
        st.code(latest.get("code_snippet", ""), language="python")
        if st.button("Approve"):
            _write_approval("approve")
            st.success("Approved. Resume the run.")
        if st.button("Reject"):
            _write_approval("reject")
            st.warning("Rejected. The run will halt.")

st.subheader("Recent Decision Traces")
for entry in reversed(logs[-20:]):
    st.json(entry)
