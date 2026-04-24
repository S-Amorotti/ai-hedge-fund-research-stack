from types import SimpleNamespace

import pytest

from app.orchestration.graph import (
    critic_node,
    executor_node,
    human_approval_node,
    planner_node,
    risk_manager_node,
    route_after_critic,
    route_after_human,
    route_after_risk,
    run_graph,
)
from app.orchestration.state import GraphState


def _state(**kwargs) -> GraphState:
    return GraphState(hypothesis="test", **kwargs)


# ── route_after_critic ────────────────────────────────────────────────────────


def test_route_critic_veto_retries_executor():
    state = _state(critic_report={"veto": True}, retry_count=0, max_retries=2)
    assert route_after_critic(state) == "executor"


def test_route_critic_no_veto_goes_to_risk():
    state = _state(critic_report={"veto": False})
    assert route_after_critic(state) == "risk"


def test_route_critic_failure_reason():
    state = _state(critic_report={}, failure_reason="Max retries exceeded after critic veto.")
    assert route_after_critic(state) == "fail"


def test_route_critic_paused():
    state = _state(critic_report={}, pause_requested=True)
    assert route_after_critic(state) == "paused"


def test_route_critic_paused_takes_priority_over_failure():
    state = _state(critic_report={}, pause_requested=True, failure_reason="some error")
    assert route_after_critic(state) == "paused"


# ── route_after_risk ──────────────────────────────────────────────────────────


def test_route_risk_goes_to_approval():
    state = _state()
    assert route_after_risk(state) == "approval"


def test_route_risk_fail():
    state = _state(failure_reason="Risk limits exceeded.")
    assert route_after_risk(state) == "fail"


def test_route_risk_paused():
    state = _state(pause_requested=True)
    assert route_after_risk(state) == "paused"


# ── route_after_human ─────────────────────────────────────────────────────────


def test_route_human_approved():
    state = _state(human_approval=True)
    assert route_after_human(state) == "done"


def test_route_human_rejected_failure():
    state = _state(failure_reason="Rejected by human approver.")
    assert route_after_human(state) == "fail"


def test_route_human_awaiting():
    state = _state(pause_requested=True)
    assert route_after_human(state) == "paused"


def test_route_human_paused_beats_approval():
    state = _state(human_approval=True, pause_requested=True)
    assert route_after_human(state) == "paused"


# ── node execution ────────────────────────────────────────────────────────────


def test_planner_node_populates_plan_and_logs(monkeypatch):
    monkeypatch.setattr(
        "app.orchestration.graph.PLANNER.plan",
        lambda hypothesis: [hypothesis, "step"],
    )
    state = _state()

    result = planner_node(state)

    assert result.active_node == "planner"
    assert result.plan == ["test", "step"]
    assert result.messages[-1]["role"] == "planner"
    assert result.logs[-1] == "Planner produced research plan"


def test_planner_node_pauses_when_flag_present(tmp_path, monkeypatch):
    pause_flag = tmp_path / "pause.flag"
    pause_flag.write_text("paused")
    monkeypatch.setattr("app.orchestration.graph.PAUSE_FLAG", str(pause_flag))
    state = _state()

    result = planner_node(state)

    assert result.failure_reason == "Paused by operator."
    assert result.active_node == ""


def test_executor_node_records_artifacts_and_compliance(monkeypatch):
    monkeypatch.setattr(
        "app.orchestration.graph.EXECUTOR.execute",
        lambda plan: {
            "code_snippet": "print('hi')",
            "market_data": {"ohlcv": [{"close": 1.0}]},
            "predictions": [1.0],
        },
    )
    monkeypatch.setattr(
        "app.orchestration.graph.COMPLIANCE.review",
        lambda symbols, trades: {"status": "pass", "symbols": symbols, "trades": trades},
    )
    state = _state(plan=["collect data"])

    result = executor_node(state)

    assert result.active_node == "executor"
    assert result.code_snippet == "print('hi')"
    assert result.market_data["ohlcv"][0]["close"] == 1.0
    assert result.compliance_report["status"] == "pass"
    assert result.logs[-2:] == [
        "Executor produced analysis artifacts",
        "Compliance review completed",
    ]


def test_critic_node_forces_veto_and_retries(monkeypatch):
    monkeypatch.setattr("app.orchestration.graph.generate_counterfactuals", lambda _: [{}, {}])
    monkeypatch.setattr(
        "app.orchestration.graph.build_consistency_report",
        lambda baseline, counterfactual_predictions: {"prediction_consistency": 0.1},
    )
    monkeypatch.setattr(
        "app.orchestration.graph.CRITIC.evaluate",
        lambda artifacts: {
            "confidence": 0.9,
            "critique_score": 0.9,
            "veto": False,
            "notes": "base note",
        },
    )
    state = _state(executor_artifacts={"predictions": [1.0]}, retry_count=0, max_retries=2)

    result = critic_node(state)

    assert result.active_node == "critic"
    assert result.critic_report["veto"] is True
    assert "PC below threshold" in result.critic_report["notes"]
    assert result.retry_count == 1
    assert result.logs[-2:] == ["Critic vetoed; retrying executor", "Critic issued report"]


def test_critic_node_sets_failure_after_max_retries(monkeypatch):
    monkeypatch.setattr("app.orchestration.graph.generate_counterfactuals", lambda _: [])
    monkeypatch.setattr(
        "app.orchestration.graph.build_consistency_report",
        lambda baseline, counterfactual_predictions: {"prediction_consistency": 1.0},
    )
    monkeypatch.setattr(
        "app.orchestration.graph.CRITIC.evaluate",
        lambda artifacts: {
            "confidence": 0.2,
            "critique_score": 0.2,
            "veto": True,
            "notes": "",
        },
    )
    state = _state(executor_artifacts={}, retry_count=2, max_retries=2)

    result = critic_node(state)

    assert result.failure_reason == "Max retries exceeded after critic veto."
    assert result.logs[-1] == "Critic issued report"


def test_risk_manager_node_sets_failure_on_veto(monkeypatch):
    monkeypatch.setattr(
        "app.orchestration.graph.RISK_MANAGER.evaluate",
        lambda artifacts: {"status": "fail", "violations": ["exposure"]},
    )
    state = _state(executor_artifacts={"risk_metrics": {"exposure": 1.5}})

    result = risk_manager_node(state)

    assert result.active_node == "risk_manager"
    assert result.failure_reason == "Risk manager vetoed based on limits."
    assert result.messages[-1]["role"] == "risk_manager"


def test_human_approval_node_approve(tmp_path, monkeypatch):
    approval_flag = tmp_path / "approval.flag"
    approval_flag.write_text("approve")
    monkeypatch.setattr("app.orchestration.graph.APPROVAL_FLAG", str(approval_flag))
    state = _state()

    result = human_approval_node(state)

    assert result.human_approval is True
    assert result.awaiting_approval is False
    assert result.messages[-1]["content"] == "Approved."
    assert result.logs[-1] == "Human approval granted"


def test_human_approval_node_reject(tmp_path, monkeypatch):
    approval_flag = tmp_path / "approval.flag"
    approval_flag.write_text("reject")
    monkeypatch.setattr("app.orchestration.graph.APPROVAL_FLAG", str(approval_flag))
    state = _state()

    result = human_approval_node(state)

    assert result.human_approval is False
    assert result.failure_reason == "Rejected by human approver."
    assert result.logs[-1] == "Human approval rejected"


def test_human_approval_node_waits_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.orchestration.graph.APPROVAL_FLAG", str(tmp_path / "missing.flag"))
    state = _state()

    result = human_approval_node(state)

    assert result.pause_requested is True
    assert result.awaiting_approval is True
    assert result.logs[-1] == "Awaiting human approval"


# ── run_graph ────────────────────────────────────────────────────────────────


def test_run_graph_logs_and_returns_state(monkeypatch):
    seen: dict[str, object] = {}

    class FakeCompiledGraph:
        def invoke(self, state: GraphState) -> GraphState:
            seen["input_state"] = state
            return state

    class FakeGraph:
        def compile(self) -> FakeCompiledGraph:
            return FakeCompiledGraph()

    class FakeLogger:
        def log_state(self, state: GraphState) -> None:
            seen["logged_state"] = state

    monkeypatch.setattr("app.orchestration.graph.build_graph", lambda: FakeGraph())
    monkeypatch.setattr("app.orchestration.graph.DecisionLogger", lambda: FakeLogger())

    result = run_graph("Hypothesis: test")

    input_state = seen["input_state"]
    assert isinstance(input_state, GraphState)
    assert input_state.messages[0] == {"role": "user", "content": "Hypothesis: test"}
    assert seen["logged_state"] is result


def test_run_graph_raises_on_failure(monkeypatch):
    class FakeCompiledGraph:
        def invoke(self, state: GraphState) -> dict[str, object]:
            return {"hypothesis": state.hypothesis, "failure_reason": "bad run"}

    class FakeGraph:
        def compile(self) -> FakeCompiledGraph:
            return FakeCompiledGraph()

    monkeypatch.setattr("app.orchestration.graph.build_graph", lambda: FakeGraph())
    monkeypatch.setattr(
        "app.orchestration.graph.DecisionLogger",
        lambda: SimpleNamespace(log_state=lambda state: None),
    )

    with pytest.raises(RuntimeError, match="bad run"):
        run_graph("Hypothesis: fail")


# ── build_graph ───────────────────────────────────────────────────────────────


def test_build_graph_compiles():
    pytest.importorskip("langgraph.graph")
    from app.orchestration.graph import build_graph

    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None
