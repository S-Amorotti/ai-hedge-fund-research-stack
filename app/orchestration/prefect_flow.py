from __future__ import annotations

import os

from prefect import flow, task, get_run_logger

from .graph import (
    planner_node,
    executor_node,
    critic_node,
    risk_manager_node,
    human_approval_node,
    route_after_critic,
    route_after_risk,
    route_after_human,
    MAX_RETRIES,
)
from .state import GraphState
from ..monitoring.log_writer import DecisionLogger


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


@flow(name="Research Orchestration")
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

        route = route_after_critic(state)
        logger.info("Routing decision: %s", route)
        if route == "executor":
            continue
        if route in {"fail", "paused"}:
            break

        state = risk_task(state)
        if state.failure_reason:
            break

        route = route_after_risk(state)
        logger.info("Risk routing decision: %s", route)
        if route in {"fail", "paused"}:
            break

        state = approval_task(state)
        if state.failure_reason:
            break

        route = route_after_human(state)
        logger.info("Approval routing decision: %s", route)
        if route in {"done", "fail", "paused"}:
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
