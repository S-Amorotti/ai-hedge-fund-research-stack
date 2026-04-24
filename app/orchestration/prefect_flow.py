from __future__ import annotations

import os

from prefect import flow, get_run_logger, task

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


@task  # type: ignore[untyped-decorator]
def planner_task(state: GraphState) -> GraphState:
    return planner_node(state)


@task  # type: ignore[untyped-decorator]
def executor_task(state: GraphState) -> GraphState:
    return executor_node(state)


@task  # type: ignore[untyped-decorator]
def critic_task(state: GraphState) -> GraphState:
    return critic_node(state)


@task  # type: ignore[untyped-decorator]
def risk_task(state: GraphState) -> GraphState:
    return risk_manager_node(state)


@task  # type: ignore[untyped-decorator]
def approval_task(state: GraphState) -> GraphState:
    return human_approval_node(state)


@flow(name="Research Orchestration")  # type: ignore[untyped-decorator]
def research_flow(hypothesis: str, max_retries: int = MAX_RETRIES) -> GraphState:
    logger = get_run_logger()
    state = GraphState(hypothesis=hypothesis, max_retries=max_retries)

    while True:
        state = planner_task(state)
        if state.failure_reason:
            break

        state = executor_task(state)
        if state.failure_reason:
            break

        state = critic_task(state)
        if state.failure_reason:
            break

        critic_route = route_after_critic(state)
        logger.info("Routing decision: %s", critic_route)
        if critic_route == "executor":
            continue
        if critic_route in {"fail", "paused"}:
            break

        state = risk_task(state)
        if state.failure_reason:
            break

        risk_route = route_after_risk(state)
        logger.info("Risk routing decision: %s", risk_route)
        if risk_route in {"fail", "paused"}:
            break

        state = approval_task(state)
        if state.failure_reason:
            break

        approval_route = route_after_human(state)
        logger.info("Approval routing decision: %s", approval_route)
        if approval_route in {"done", "fail", "paused"}:
            break

    DecisionLogger().log_state(state)

    if state.failure_reason:
        raise RuntimeError(state.failure_reason)

    return state


if __name__ == "__main__":
    hypothesis = os.getenv(
        "HYPOTHESIS",
        "Hypothesis: earnings sentiment predicts short-term drift",
    )
    research_flow(hypothesis=hypothesis)
