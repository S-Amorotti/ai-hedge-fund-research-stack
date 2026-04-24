from __future__ import annotations

from collections.abc import Callable
import os
from typing import Protocol, TypeVar

try:
    from prefect import flow, get_run_logger, task
except ImportError:
    F = TypeVar("F", bound=Callable[..., object])

    class _NoOpLogger:
        def info(self, message: str, *args: object) -> None:
            del message, args

    def task(func: F | None = None, **_kwargs: object) -> F | Callable[[F], F]:
        if func is None:
            def decorator(inner: F) -> F:
                return inner

            return decorator
        return func

    def flow(*_args: object, **_kwargs: object) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator

    def get_run_logger() -> _NoOpLogger:
        return _NoOpLogger()

from ..monitoring.log_writer import DecisionLogger
from .graph import (
    MAX_RETRIES,
    critic_node,
    executor_node,
    human_approval_node,
    planner_node,
    risk_manager_node,
    route_after_critic,
    route_after_human,
    route_after_risk,
)
from .state import GraphState


class SupportsInfo(Protocol):
    def info(self, message: str, *args: object) -> None: ...


class SupportsLogState(Protocol):
    def log_state(self, state: GraphState) -> None: ...


StateRunner = Callable[[GraphState], GraphState]


@task
def planner_task(state: GraphState) -> GraphState:
    return planner_node(state)


@task
def executor_task(state: GraphState) -> GraphState:
    return executor_node(state)


@task
def critic_task(state: GraphState) -> GraphState:
    return critic_node(state)


@task
def risk_task(state: GraphState) -> GraphState:
    return risk_manager_node(state)


@task
def approval_task(state: GraphState) -> GraphState:
    return human_approval_node(state)


def _run_research_loop(
    hypothesis: str,
    max_retries: int = MAX_RETRIES,
    planner_runner: StateRunner = planner_task,
    executor_runner: StateRunner = executor_task,
    critic_runner: StateRunner = critic_task,
    risk_runner: StateRunner = risk_task,
    approval_runner: StateRunner = approval_task,
    route_logger: SupportsInfo | None = None,
    state_logger: SupportsLogState | None = None,
) -> GraphState:
    logger = route_logger or get_run_logger()
    state = GraphState(hypothesis=hypothesis, max_retries=max_retries)

    while True:
        state = planner_runner(state)
        if state.failure_reason:
            break

        state = executor_runner(state)
        if state.failure_reason:
            break

        state = critic_runner(state)
        if state.failure_reason:
            break

        critic_route = route_after_critic(state)
        logger.info("Routing decision: %s", critic_route)
        if critic_route == "executor":
            continue
        if critic_route in {"fail", "paused"}:
            break

        state = risk_runner(state)
        if state.failure_reason:
            break

        risk_route = route_after_risk(state)
        logger.info("Risk routing decision: %s", risk_route)
        if risk_route in {"fail", "paused"}:
            break

        state = approval_runner(state)
        if state.failure_reason:
            break

        approval_route = route_after_human(state)
        logger.info("Approval routing decision: %s", approval_route)
        if approval_route in {"done", "fail", "paused"}:
            break

    state_logger = state_logger or DecisionLogger()
    state_logger.log_state(state)

    if state.failure_reason:
        raise RuntimeError(state.failure_reason)

    return state


@flow(name="Research Orchestration")
def research_flow(hypothesis: str, max_retries: int = MAX_RETRIES) -> GraphState:
    return _run_research_loop(hypothesis=hypothesis, max_retries=max_retries)


if __name__ == "__main__":
    hypothesis = os.getenv(
        "HYPOTHESIS",
        "Hypothesis: earnings sentiment predicts short-term drift",
    )
    research_flow(hypothesis=hypothesis)
