from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from ..orchestration.state import GraphState
from ..memory.memory_manager import store_trace


class DecisionLogger:
    """Append-only JSONL logger for auditability."""

    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = log_path or os.getenv(
            "LOG_PATH", "app/monitoring/decisions.log"
        )

    def log_state(self, state: GraphState | dict[str, Any]) -> None:
        if isinstance(state, dict):
            state = GraphState(**state)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": state.hypothesis,
            "messages": state.messages,
            "market_data": state.market_data,
            "code_snippet": state.code_snippet,
            "critique_score": state.critique_score,
            "human_approval": state.human_approval,
            "awaiting_approval": state.awaiting_approval,
            "plan": state.plan,
            "executor_artifacts": state.executor_artifacts,
            "critic_report": state.critic_report,
            "compliance_report": state.compliance_report,
            "risk_report": state.risk_report,
            "retry_count": state.retry_count,
            "max_retries": state.max_retries,
            "pause_requested": state.pause_requested,
            "confidence": state.confidence,
            "failure_reason": state.failure_reason,
            "logs": state.logs,
            "active_node": state.active_node,
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        store_trace(record)


def read_logs(log_path: str | None = None) -> list[dict[str, Any]]:
    path = log_path or os.getenv("LOG_PATH", "app/monitoring/decisions.log")
    if not os.path.exists(path):
        return []

    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries
