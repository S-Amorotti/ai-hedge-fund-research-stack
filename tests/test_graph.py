from app.orchestration.graph import (
    route_after_critic,
    route_after_human,
    route_after_risk,
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


# ── build_graph ───────────────────────────────────────────────────────────────


def test_build_graph_compiles():
    from app.orchestration.graph import build_graph

    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None
