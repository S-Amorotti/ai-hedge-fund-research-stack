import pytest

from app.orchestration.agents import COMPLIANCE, CRITIC, PLANNER, RISK_MANAGER
from app.orchestration.tools import ToolSafetyError


def test_planner_plan_structure():
    plan = PLANNER.plan("Hypothesis: momentum predicts returns")
    assert len(plan) == 4
    assert "Hypothesis: momentum predicts returns" in plan[0]


def test_planner_has_no_tool_access():
    with pytest.raises(ToolSafetyError):
        PLANNER.validate_tool_access("fetch_market_data")


def test_critic_no_lookahead_bias():
    report = CRITIC.evaluate({"code_snippet": "data = df.copy()"})
    assert report["look_ahead_bias"] == "low"
    assert report["veto"] is False
    assert report["critique_score"] >= 0.8


def test_critic_detects_shift_lookahead():
    report = CRITIC.evaluate({"code_snippet": "x = df['close'].shift(-1)"})
    assert report["look_ahead_bias"] == "high"
    assert report["veto"] is True
    assert report["critique_score"] < 0.8


def test_critic_detects_future_keyword():
    report = CRITIC.evaluate({"code_snippet": "future_price = df['close'].mean()"})
    assert report["look_ahead_bias"] == "high"


def test_critic_empty_snippet():
    report = CRITIC.evaluate({"code_snippet": ""})
    assert "veto" in report
    assert "critique_score" in report


def test_risk_manager_pass():
    report = RISK_MANAGER.evaluate({"risk_metrics": {"max_drawdown": 0.05, "exposure": 0.8}})
    assert report["status"] == "pass"
    assert report["violations"] == []


def test_risk_manager_fail_drawdown():
    report = RISK_MANAGER.evaluate({"risk_metrics": {"max_drawdown": 0.25, "exposure": 0.5}})
    assert report["status"] == "fail"
    assert "max_drawdown" in report["violations"]


def test_risk_manager_fail_exposure():
    report = RISK_MANAGER.evaluate({"risk_metrics": {"max_drawdown": 0.1, "exposure": 1.5}})
    assert report["status"] == "fail"
    assert "exposure" in report["violations"]


def test_risk_manager_fail_both():
    report = RISK_MANAGER.evaluate({"risk_metrics": {"max_drawdown": 0.5, "exposure": 2.0}})
    assert report["status"] == "fail"
    assert "max_drawdown" in report["violations"]
    assert "exposure" in report["violations"]


def test_risk_manager_missing_metrics():
    report = RISK_MANAGER.evaluate({})
    assert report["status"] == "pass"
    assert report["metrics"]["max_drawdown"] == 0.0


def test_compliance_empty_inputs():
    report = COMPLIANCE.review(symbols=[], trades=[])
    assert report["status"] == "pass"
    assert report["symbol_report"]["status"] == "pass"
    assert report["wash_sale_report"]["status"] == "pass"


def test_compliance_with_trades_fails():
    report = COMPLIANCE.review(symbols=[], trades=[{"symbol": "AAPL", "qty": 100}])
    assert report["wash_sale_report"]["status"] == "fail"
    assert report["status"] == "fail"


def test_executor_tool_access_denied():
    from app.orchestration.agents import CRITIC

    with pytest.raises(ToolSafetyError):
        CRITIC.validate_tool_access("fetch_market_data")
