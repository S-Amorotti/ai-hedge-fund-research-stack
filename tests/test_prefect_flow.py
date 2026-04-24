from __future__ import annotations

from app.orchestration.prefect_flow import _run_research_loop
from app.orchestration.state import GraphState


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        rendered = message % args if args else message
        self.messages.append(rendered)


class FakeStateLogger:
    def __init__(self) -> None:
        self.states: list[GraphState] = []

    def log_state(self, state: GraphState) -> None:
        self.states.append(state)


def test_run_research_loop_success_path_logs_final_state():
    route_logger = FakeLogger()
    state_logger = FakeStateLogger()

    def planner(state: GraphState) -> GraphState:
        state.plan = ["step"]
        return state

    def executor(state: GraphState) -> GraphState:
        state.executor_artifacts = {}
        return state

    def critic(state: GraphState) -> GraphState:
        state.critic_report = {"veto": False}
        return state

    def risk(state: GraphState) -> GraphState:
        return state

    def approval(state: GraphState) -> GraphState:
        state.human_approval = True
        return state

    result = _run_research_loop(
        hypothesis="Hypothesis: success",
        planner_runner=planner,
        executor_runner=executor,
        critic_runner=critic,
        risk_runner=risk,
        approval_runner=approval,
        route_logger=route_logger,
        state_logger=state_logger,
    )

    assert result.hypothesis == "Hypothesis: success"
    assert state_logger.states[-1] is result
    assert route_logger.messages == [
        "Routing decision: risk",
        "Risk routing decision: approval",
        "Approval routing decision: done",
    ]


def test_run_research_loop_retries_executor_then_completes():
    route_logger = FakeLogger()
    state_logger = FakeStateLogger()
    counts = {"planner": 0, "executor": 0, "critic": 0}

    def planner(state: GraphState) -> GraphState:
        counts["planner"] += 1
        return state

    def executor(state: GraphState) -> GraphState:
        counts["executor"] += 1
        return state

    def critic(state: GraphState) -> GraphState:
        counts["critic"] += 1
        state.critic_report = {"veto": counts["critic"] == 1}
        return state

    def risk(state: GraphState) -> GraphState:
        return state

    def approval(state: GraphState) -> GraphState:
        state.human_approval = True
        return state

    _run_research_loop(
        hypothesis="Hypothesis: retry",
        planner_runner=planner,
        executor_runner=executor,
        critic_runner=critic,
        risk_runner=risk,
        approval_runner=approval,
        route_logger=route_logger,
        state_logger=state_logger,
    )

    assert counts == {"planner": 2, "executor": 2, "critic": 2}
    assert route_logger.messages[0] == "Routing decision: executor"
    assert route_logger.messages[-1] == "Approval routing decision: done"


def test_run_research_loop_raises_and_still_logs_state():
    route_logger = FakeLogger()
    state_logger = FakeStateLogger()

    def planner(state: GraphState) -> GraphState:
        state.failure_reason = "planner failed"
        return state

    try:
        _run_research_loop(
            hypothesis="Hypothesis: failure",
            planner_runner=planner,
            route_logger=route_logger,
            state_logger=state_logger,
        )
    except RuntimeError as exc:
        assert str(exc) == "planner failed"
    else:
        raise AssertionError("expected RuntimeError")

    assert state_logger.states[-1].failure_reason == "planner failed"
    assert route_logger.messages == []
