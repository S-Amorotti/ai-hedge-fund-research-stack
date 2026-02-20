from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class GraphState:
    """Shared state for the LangGraph workflow.

    The state is intentionally explicit and auditable. Any new fields must be
    documented here to preserve reproducibility and reviewability.
    """

    hypothesis: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    market_data: Dict[str, Any] = field(default_factory=dict)
    code_snippet: str = ""
    critique_score: float = 0.0
    human_approval: bool = False
    awaiting_approval: bool = False
    plan: List[str] = field(default_factory=list)
    executor_artifacts: Dict[str, Any] = field(default_factory=dict)
    critic_report: Dict[str, Any] = field(default_factory=dict)
    compliance_report: Dict[str, Any] = field(default_factory=dict)
    risk_report: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 2
    pause_requested: bool = False
    confidence: float = 0.0
    logs: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    active_node: str = ""

    def log(self, message: str) -> None:
        self.logs.append(message)
