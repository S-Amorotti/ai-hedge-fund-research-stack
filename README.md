# AI Hedge Fund (Research Stack Only)

This repository implements a **research-only** AI hedge fund infrastructure with strong safety controls. It does **not** place trades, connect to brokerages, or execute orders. Every decision is logged, reproducible, and fail-closed.

**Non-negotiable guarantees**
- No live trading, no order placement, no broker APIs.
- All agents are role-limited with explicit tool allowlists.
- Critic agent has hard veto power.
- Human-in-the-loop pause/kill switch is enforced.
- Every decision is logged and retrievable.
- Deterministic defaults for evaluators.
- Fail-closed on ambiguity or error.

## System Architecture

**Orchestration (LangGraph DAG)**
- **Planner**: decomposes a hypothesis into research steps. No data access.
- **Executor (SCG)**: generates executable research code only. Does not run it.
- **Critic**: evaluates for look-ahead bias, overfitting, leakage, and reproducibility. Can veto.
- **Risk Manager**: enforces max drawdown and exposure limits.
- **Human Approval**: HITL gate to approve or reject runs.
- **Compliance Officer**: checks restricted symbols and wash-sale patterns (informational).

Graph: `Planner → Executor → Critic → Risk Manager → Human Approval`
- Critic veto routes back to `Executor`.
- Max retries enforced.
- On retry exhaustion, the system fails closed.

**Validation (Counterfactual Testing)**
- Generates 50 counterfactual datasets per run.
- Perturbations include price noise, earnings-date shifts (±3 days), and inverted sentiment.
- Computes **Prediction Consistency (PC)** in [0,1].
- Low PC blocks downstream approval.

**Memory Layer (Postgres + pgvector)**
- Stores reasoning traces, summaries, and embeddings.
- Raw traces are summarized before embedding.
- Retrieval uses cosine similarity.
- Past failures are retrievable before decision (RAG).

**Monitoring & Human-in-the-Loop**
- Streamlit dashboard shows real-time traces, confidence meter, and pause/resume controls.
- Alert when confidence < 70%.
- Kill switch halts LangGraph execution safely with no state corruption.
- Prefect orchestration provides run history, logs, and visual flow monitoring.

## Repository Layout
```
ai-hedge-fund/
  app/
    orchestration/
      graph.py
      prefect_flow.py
      state.py
      agents.py
      tools.py
    validation/
      counterfactual.py
      metrics.py
    memory/
      db.py
      memory_manager.py
      schema.sql
    monitoring/
      dashboard.py
      log_writer.py
  infra/
    docker-compose.yml
  tests/
  pyproject.toml
  README.md
```

## Design Principles

**Fail-closed by default**
- If a tool is not explicitly configured and audited, it fails with a safety error.
- If the Critic vetoes and retries are exhausted, execution stops with an error.

**No “god agent”**
- Each agent is role-limited with explicit allowlisted tools.
- Agents cannot access tools outside their allowlist.

**Auditability and reproducibility**
- All decisions are written to JSONL logs and persisted to Postgres.
- Deterministic defaults (fixed seeds and low temperatures for evaluators).

**Human-in-the-loop controls**
- Pause/kill switch is enforced at every node via a shared flag.
- Execution halts safely without state corruption.

## Orchestration Details

**State model**
- The graph uses a typed `GraphState` (`app/orchestration/state.py`).
- Fields include `messages`, `market_data`, `code_snippet`, `critique_score`, `human_approval`, and `risk_report` in addition to planning and review artifacts.

**Routing**
- The routing logic in `app/orchestration/graph.py` is explicit and readable.
- If `pause_requested` is true or the pause flag file exists, the system halts safely.

**Veto behavior**
- The Critic returns a structured JSON report with `veto` set to true/false.
- Veto routes back to `Executor` up to `MAX_RETRIES`.
- Exhausted retries cause a fail-closed termination.

## Agents and Tool Allowlists

**Planner**
- Purpose: research decomposition only.
- Allowed tools: none.
- File: `app/orchestration/agents.py`.

**Executor (Data Engineer)**
- Purpose: fetch/clean/analyze data only.
- Allowed tools: `fetch_market_data`, `clean_data`, `run_analysis`.
- No evaluation, no trading.

**Critic (Risk Manager)**
- Purpose: bias/leakage/reproducibility assessment.
- Allowed tools: none.
- Must return structured JSON and can veto.

**Compliance Officer**
- Purpose: restricted symbols + wash-sale pattern checks.
- Allowed tools: `check_restricted_symbols`, `check_wash_sale_patterns`.
- Cannot modify analysis.

## Counterfactual Validation

The counterfactual engine produces 50 datasets per run and re-evaluates predictions under:
- Price noise
- Earnings date shifts (±3 days)
- Inverted sentiment

**Prediction Consistency (PC)**
- Computed as the average sign agreement between baseline predictions and counterfactual predictions.
- Range is [0,1].
- If PC < `PC_THRESHOLD` (default 0.7), the Critic veto is enforced regardless of other checks.

## Memory and Retrieval (RAG)

**What is stored**
- Hypothesis
- Full decision trace (JSON)
- Summary of the trace
- Embedding of the summary
- Failure reason (if any)

**Why summaries before embeddings**
- Keeps embeddings stable, reduces noise, and improves retrieval relevance.

**Retrieval**
- Query embeddings are compared via cosine similarity against stored traces.
- Prior failures can be retrieved before new decisions.

## Monitoring & Kill Switch

**Streamlit dashboard**
- 3-column layout with controls, flow visualization, and market view.
- Confidence gauge turns red below 70%.
- Pause/resume buttons control `app/monitoring/pause.flag`.
- Human approval modal writes to `app/monitoring/approval.flag`.
- Live refresh uses Streamlit autorefresh (every 2s).

**Prefect orchestration (visual monitoring)**
- Prefect UI shows flow runs, task logs, retries, and execution timelines.
- Use the Prefect flow in `app/orchestration/prefect_flow.py` to run with orchestration.

**Kill switch behavior**
- The graph checks the pause flag at every node.
- If paused, execution halts and records the failure reason.

## Infrastructure

**Local-first**
- Uses Postgres + pgvector via Docker Compose.
- No cloud dependencies.

**Docker Compose**
- File: `infra/docker-compose.yml`.
- Exposes Postgres on port 5432.

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string.
- `LOG_PATH`: Optional path for JSONL logs (default `app/monitoring/decisions.log`).
- `MAX_RETRIES`: Max executor retries after critic veto (default 2).
- `PC_THRESHOLD`: Minimum Prediction Consistency threshold (default 0.7).
- `PAUSE_FLAG`: Path for pause flag file (default `app/monitoring/pause.flag`).

## Quickstart

1. Start Postgres + pgvector (host port 5433):
   ```bash
   cd ai-hedge-fund
   docker compose -f infra/docker-compose.yml up
   ```

2. Initialize the schema (note the port):
   ```bash
   export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ai_hedge_fund
   psql "$DATABASE_URL" -f app/memory/schema.sql
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the dashboard:
   ```bash
   streamlit run app/monitoring/dashboard.py
   ```

3. Install dependencies with `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Launch the dashboard:
   ```bash
   uv run streamlit run app/monitoring/dashboard.py
   ```

5. (Optional) Start Prefect server + UI:
   ```bash
   PREFECT_SERVER_ANALYTICS_ENABLED=false uv run prefect server start
   ```
   If the CLI does not connect to the server, set:
   ```bash
   uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   ```
   To opt out of client analytics, set:
   ```bash
   DO_NOT_TRACK=1
   ```

6. Run a research flow (example):
   ```bash
   uv run python -c "from app.orchestration.graph import run_graph; run_graph('Hypothesis: earnings sentiment predicts short-term drift')"
   ```

7. Run the Prefect-orchestrated flow (with visual monitoring):
   ```bash
   uv run python -m app.orchestration.prefect_flow
   ```

## Safety Boundaries (Explicit)

- No broker connections are permitted.
- No order placement is implemented or allowed.
- Tool calls are strictly allowlisted per agent.
- External data access is disabled unless explicitly configured and audited.

## Extending the System Safely

When you add new components:
- Keep agents role-limited.
- Add any new tool to `app/orchestration/tools.py` and update allowlists explicitly.
- Preserve fail-closed behavior on any error or ambiguity.
- Ensure new logic is logged and auditable.

## FAQ

**Q: Why is the Critic allowed to veto?**
- The Critic mimics institutional risk controls. It prevents bias, leakage, and non-reproducible signals from proceeding.

**Q: Why require counterfactual testing?**
- It detects profit mirages and spurious correlations by stress-testing signals.

**Q: Can this system be connected to a broker?**
- No. This is prohibited by design and policy.

## Compliance Note

This project is designed as if it will be audited. All decisions are reproducible, logged, and fail-closed. Any change that risks trading, privilege escalation, or hidden tool access must be rejected.
