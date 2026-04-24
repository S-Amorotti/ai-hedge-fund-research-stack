import uuid

import pytest

from app.memory.db import get_connection
from app.memory.memory_manager import retrieve_similar, store_trace

pytestmark = pytest.mark.integration


def _trace(hypothesis: str) -> dict[str, object]:
    return {
        "hypothesis": hypothesis,
        "messages": [],
        "market_data": {},
        "code_snippet": "print('integration')",
        "critique_score": 0.0,
        "human_approval": False,
        "awaiting_approval": False,
        "plan": ["integration smoke test"],
        "executor_artifacts": {},
        "critic_report": {},
        "compliance_report": {},
        "risk_report": {},
        "retry_count": 0,
        "max_retries": 2,
        "pause_requested": False,
        "confidence": 0.0,
        "failure_reason": "integration smoke test",
        "logs": [],
        "active_node": "integration",
    }


def test_store_trace_and_retrieve_similar_round_trip() -> None:
    hypothesis = f"integration-{uuid.uuid4()}"
    trace = _trace(hypothesis)

    try:
        store_trace(trace)
        matches = retrieve_similar("integration smoke test", limit=10)
        assert any(match["hypothesis"] == hypothesis for match in matches)
    finally:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM decision_traces WHERE hypothesis = %s", (hypothesis,))
            conn.commit()
