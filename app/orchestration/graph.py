from __future__ import annotations

import os
from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, END

from .state import GraphState
from .agents import PLANNER, EXECUTOR, CRITIC, COMPLIANCE, RISK_MANAGER
from ..validation.counterfactual import generate_counterfactuals
from ..validation.metrics import build_consistency_report
from ..monitoring.log_writer import DecisionLogger


MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
PC_THRESHOLD = float(os.getenv("PC_THRESHOLD", "0.7"))
PAUSE_FLAG = os.getenv("PAUSE_FLAG", "app/monitoring/pause.flag")
APPROVAL_FLAG = os.getenv("APPROVAL_FLAG", "app/monitoring/approval.flag")


def _ensure_not_paused(state: GraphState) -> GraphState:
    if state.pause_requested or os.path.exists(PAUSE_FLAG):
        state.failure_reason = "Paused by operator."
        return state
    return state


def planner_node(state: GraphState) -> GraphState:
    state = _ensure_not_paused(state)
    if state.failure_reason:
        return state

    state.active_node = "planner"
    state.plan = PLANNER.plan(state.hypothesis)
    state.messages.append({"role": "planner", "content": "Plan created."})
    state.log("Planner produced research plan")
    return state


def executor_node(state: GraphState) -> GraphState:
    state = _ensure_not_paused(state)
    if state.failure_reason:
        return state

    state.active_node = "executor"
    state.executor_artifacts = EXECUTOR.execute(state.plan)
    state.code_snippet = state.executor_artifacts.get("code_snippet", "")
    state.market_data = state.executor_artifacts.get("market_data", {})
    state.messages.append({"role": "executor", "content": "Code generated (not executed)."})
    state.log("Executor produced analysis artifacts")

    # Compliance review is informational and cannot modify analysis.
    state.compliance_report = COMPLIANCE.review(symbols=[], trades=[])
    state.log("Compliance review completed")
    return state


def critic_node(state: GraphState) -> GraphState:
    state = _ensure_not_paused(state)
    if state.failure_reason:
        return state

    state.active_node = "critic"
    # Counterfactual validation of predictions (stubbed inputs for now).
    baseline_predictions = state.executor_artifacts.get("predictions", [])
    counterfactuals = generate_counterfactuals({})
    cf_predictions = []
    for _ in counterfactuals:
        cf_predictions.append([])

    consistency_report = build_consistency_report(
        baseline=_to_array(baseline_predictions),
        counterfactual_predictions=[_to_array(p) for p in cf_predictions],
    )

    report = CRITIC.evaluate(state.executor_artifacts)
    report["counterfactual"] = consistency_report
    state.critic_report = report
    state.confidence = float(report.get("confidence", 0.0))
    state.critique_score = float(report.get("critique_score", state.confidence))
    state.messages.append({"role": "critic", "content": "Critic review complete."})

    if consistency_report["prediction_consistency"] < PC_THRESHOLD:
        state.critic_report["veto"] = True
        state.critic_report["notes"] = (
            state.critic_report.get("notes", "")
            + f" PC below threshold {PC_THRESHOLD}."
        )

    if state.critic_report.get("veto", False):
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            state.log("Critic vetoed; retrying executor")
        else:
            state.failure_reason = "Max retries exceeded after critic veto."

    state.log("Critic issued report")
    return state


def risk_manager_node(state: GraphState) -> GraphState:
    state = _ensure_not_paused(state)
    if state.failure_reason:
        return state

    state.active_node = "risk_manager"
    state.risk_report = RISK_MANAGER.evaluate(state.executor_artifacts)
    state.messages.append({"role": "risk_manager", "content": "Risk checks complete."})
    state.log("Risk manager issued report")
    if state.risk_report.get("status") != "pass":
        state.failure_reason = "Risk manager vetoed based on limits."
    return state


def human_approval_node(state: GraphState) -> GraphState:
    state = _ensure_not_paused(state)
    if state.failure_reason:
        return state

    state.active_node = "human_approval"
    state.awaiting_approval = True
    approval_value = None
    if os.path.exists(APPROVAL_FLAG):
        with open(APPROVAL_FLAG, "r", encoding="utf-8") as handle:
            approval_value = handle.read().strip().lower()

    if approval_value == "approve":
        state.human_approval = True
        state.awaiting_approval = False
        state.messages.append({"role": "human", "content": "Approved."})
        state.log("Human approval granted")
    elif approval_value == "reject":
        state.human_approval = False
        state.awaiting_approval = False
        state.failure_reason = "Rejected by human approver."
        state.messages.append({"role": "human", "content": "Rejected."})
        state.log("Human approval rejected")
    else:
        state.pause_requested = True
        state.log("Awaiting human approval")

    return state


def _to_array(values: Any):
    import numpy as np

    return np.asarray(values, dtype=float)


def route_after_critic(state: GraphState) -> Literal["executor", "risk", "fail", "paused"]:
    if state.pause_requested:
        return "paused"

    if state.failure_reason:
        return "fail"

    veto = bool(state.critic_report.get("veto", True))
    if veto:
        return "executor"

    return "risk"


def route_after_risk(state: GraphState) -> Literal["approval", "fail", "paused"]:
    if state.pause_requested:
        return "paused"
    if state.failure_reason:
        return "fail"
    return "approval"


def route_after_human(state: GraphState) -> Literal["done", "fail", "paused"]:
    if state.pause_requested:
        return "paused"
    if state.failure_reason:
        return "fail"
    if state.human_approval:
        return "done"
    return "paused"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("risk", risk_manager_node)
    graph.add_node("approval", human_approval_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "executor": "executor",
            "risk": "risk",
            "fail": END,
            "paused": END,
        },
    )

    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {
            "approval": "approval",
            "fail": END,
            "paused": END,
        },
    )

    graph.add_conditional_edges(
        "approval",
        route_after_human,
        {
            "done": END,
            "fail": END,
            "paused": END,
        },
    )

    return graph


def run_graph(hypothesis: str) -> GraphState:
    state = GraphState(hypothesis=hypothesis, max_retries=MAX_RETRIES)
    state.messages.append({"role": "user", "content": hypothesis})
    graph = build_graph().compile()
    logger = DecisionLogger()

    final_state = graph.invoke(state)
    if isinstance(final_state, dict):
        final_state = GraphState(**final_state)
    logger.log_state(final_state)

    if final_state.failure_reason:
        raise RuntimeError(final_state.failure_reason)

    return final_state
