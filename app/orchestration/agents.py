from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List

from . import tools


@dataclass(frozen=True)
class AgentConfig:
    name: str
    system_prompt: str
    allowed_tools: List[str]
    temperature: float


class BaseAgent:
    config: AgentConfig

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def validate_tool_access(self, tool_name: str) -> None:
        if tool_name not in self.config.allowed_tools:
            raise tools.ToolSafetyError(
                f"Tool '{tool_name}' not allowed for agent {self.config.name}."
            )


class PlannerAgent(BaseAgent):
    def plan(self, hypothesis: str) -> List[str]:
        # Deterministic, non-data-access decomposition.
        return [
            f"Restate hypothesis: {hypothesis}",
            "Identify required datasets and constraints",
            "Outline feature engineering steps",
            "Define evaluation protocol and bias checks",
        ]


class ExecutorAgent(BaseAgent):
    def execute(self, plan: List[str]) -> Dict[str, Any]:
        # Executor generates executable Python code but does not run it.
        artifacts: Dict[str, Any] = {"plan": plan}
        code_snippet = "\n".join(
            [
                "import pandas as pd",
                "import yfinance as yf",
                "",
                "def run_research(symbol: str, start: str, end: str) -> pd.DataFrame:",
                "    data = yf.download(symbol, start=start, end=end)",
                "    data['rsi'] = compute_rsi(data['Close'])",
                "    # TODO: integrate news sentiment safely via approved source",
                "    return data",
                "",
                "def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:",
                "    delta = series.diff()",
                "    gain = delta.clip(lower=0)",
                "    loss = -delta.clip(upper=0)",
                "    avg_gain = gain.rolling(period).mean()",
                "    avg_loss = loss.rolling(period).mean()",
                "    rs = avg_gain / avg_loss",
                "    return 100 - (100 / (1 + rs))",
            ]
        )
        artifacts["code_snippet"] = code_snippet
        market_request = {
            "symbol": "AAPL",
            "start": "2022-01-01",
            "end": "2022-12-31",
        }
        artifacts["market_data_request"] = market_request

        # Read-only data access + analysis for dashboard visibility.
        self.validate_tool_access("fetch_market_data")
        market_data = tools.fetch_market_data(market_request)
        self.validate_tool_access("clean_data")
        cleaned = tools.clean_data(market_data)
        self.validate_tool_access("run_analysis")
        analysis = tools.run_analysis(cleaned)
        artifacts["market_data"] = market_data | {"signals": analysis.get("signals", [])}
        artifacts["analysis"] = analysis.get("analysis", {})
        artifacts["risk_metrics"] = analysis.get("risk_metrics", {})
        return artifacts


class CriticAgent(BaseAgent):
    def evaluate(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        # Returns a structured JSON report and can veto.
        code_snippet = artifacts.get("code_snippet", "")
        look_ahead_risk = "low"
        if "shift(-" in code_snippet or "future" in code_snippet.lower():
            look_ahead_risk = "high"

        critique_score = 0.9 if look_ahead_risk == "low" else 0.4
        veto = critique_score < 0.8

        report = {
            "look_ahead_bias": look_ahead_risk,
            "overfitting": "unknown",
            "leakage": "unknown",
            "reproducibility": "unknown",
            "veto": veto,
            "confidence": critique_score,
            "critique_score": critique_score,
            "notes": "Heuristic review completed.",
        }
        return json.loads(json.dumps(report))


class ComplianceOfficerAgent(BaseAgent):
    def review(self, symbols: List[str], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.validate_tool_access("check_restricted_symbols")
        symbol_report = tools.check_restricted_symbols(symbols)
        self.validate_tool_access("check_wash_sale_patterns")
        wash_report = tools.check_wash_sale_patterns(trades)

        status = "pass" if symbol_report["status"] == "pass" and wash_report["status"] == "pass" else "fail"
        return {
            "symbol_report": symbol_report,
            "wash_sale_report": wash_report,
            "status": status,
        }


class RiskManagerAgent(BaseAgent):
    def evaluate(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        risk_metrics = artifacts.get("risk_metrics", {})
        max_drawdown = float(risk_metrics.get("max_drawdown", 0.0))
        exposure = float(risk_metrics.get("exposure", 0.0))

        limits = {
            "max_drawdown": 0.2,
            "exposure": 1.0,
        }

        violations = []
        if max_drawdown > limits["max_drawdown"]:
            violations.append("max_drawdown")
        if exposure > limits["exposure"]:
            violations.append("exposure")

        status = "pass" if not violations else "fail"
        return {
            "status": status,
            "violations": violations,
            "metrics": {"max_drawdown": max_drawdown, "exposure": exposure},
            "limits": limits,
        }


PLANNER = PlannerAgent(
    AgentConfig(
        name="Planner",
        system_prompt="You decompose hypotheses into research steps. No data access.",
        allowed_tools=[],
        temperature=0.2,
    )
)

EXECUTOR = ExecutorAgent(
    AgentConfig(
        name="Executor",
        system_prompt="You fetch and clean data, and run analysis only. No evaluation or trading.",
        allowed_tools=["clean_data", "run_analysis", "fetch_market_data"],
        temperature=0.2,
    )
)

CRITIC = CriticAgent(
    AgentConfig(
        name="Critic",
        system_prompt="You evaluate for bias, leakage, and reproducibility. You can veto.",
        allowed_tools=[],
        temperature=0.1,
    )
)

COMPLIANCE = ComplianceOfficerAgent(
    AgentConfig(
        name="ComplianceOfficer",
        system_prompt="You check restricted symbols and wash-sale patterns. No modifications allowed.",
        allowed_tools=["check_restricted_symbols", "check_wash_sale_patterns"],
        temperature=0.1,
    )
)

RISK_MANAGER = RiskManagerAgent(
    AgentConfig(
        name="RiskManager",
        system_prompt="You enforce max drawdown and exposure limits.",
        allowed_tools=[],
        temperature=0.1,
    )
)
