# Contributing

Thank you for your interest in contributing. This project is a research-only AI infrastructure stack — no trading, no order placement, no broker APIs. All contributions must preserve those guarantees.

## Development setup

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/<your-org>/ai-hedge-fund-research-stack.git
   cd ai-hedge-fund-research-stack
   ```

2. **Copy the environment template**
   ```bash
   cp .env.example .env
   # edit .env — at minimum set POSTGRES_PASSWORD and DATABASE_URL
   ```

3. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/))
   ```bash
   uv sync --dev
   ```

4. **Start the database**
   ```bash
   docker compose -f infra/docker-compose.yml up -d
   psql "$DATABASE_URL" -f app/memory/schema.sql
   ```

## Running checks

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy app

# Tests
uv run pytest
```

All four must pass before opening a pull request. CI enforces the same checks automatically.

## Pull request guidelines

- **One concern per PR.** Bug fixes, features, and refactors should be separate.
- **Tests are required.** Any new logic needs a corresponding test in `tests/`. The CI gate enforces 60 % coverage as a minimum.
- **Preserve the safety invariants.** Every change must keep the system fail-closed:
  - No broker connections.
  - No order placement.
  - Agents may only access tools on their explicit allowlist.
  - The kill switch must halt execution at the next node without state corruption.
- **Keep agents role-limited.** If you add a new tool, register it in `app/orchestration/tools.py` and update only the agent allowlists that genuinely need it.
- **Log and audit new decisions.** Any new node or agent action that produces a research output should be recorded via `DecisionLogger`.

## Adding a new agent

1. Define an `AgentConfig` with a minimal `allowed_tools` list.
2. Subclass `BaseAgent` and implement the agent's method.
3. Register the agent instance at the bottom of `agents.py`.
4. Wire it into the graph in `graph.py` — add the node, edges, and a routing function.
5. Add unit tests covering the routing logic and the agent method.

## Reporting issues

Open a GitHub issue with:
- A short description of the bug or feature request.
- Steps to reproduce (for bugs).
- The expected vs. actual behaviour.

## Code of conduct

Be respectful and constructive. This is a research project with real safety implications — treat every review comment with that context in mind.
