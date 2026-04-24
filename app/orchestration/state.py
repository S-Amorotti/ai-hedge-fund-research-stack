from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphState:
    """Shared state for the LangGraph workflow.

    The state is intentionally explicit and auditable. Any new fields must be
    documented here to preserve reproducibility and reviewability.
    """

    hypothesis: str
    messages: list[dict[str, str]] = field(default_factory=list)
    market_data: dict[str, Any] = field(default_factory=dict)
    code_snippet: str = ""
    critique_score: float = 0.0
    human_approval: bool = False
    awaiting_approval: bool = False
    plan: list[str] = field(default_factory=list)
    executor_artifacts: dict[str, Any] = field(default_factory=dict)
    critic_report: dict[str, Any] = field(default_factory=dict)
    compliance_report: dict[str, Any] = field(default_factory=dict)
    risk_report: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 2
    pause_requested: bool = False
    confidence: float = 0.0
    logs: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    active_node: str = ""

    def log(self, message: str) -> None:
        self.logs.append(message)
