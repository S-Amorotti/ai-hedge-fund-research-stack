from __future__ import annotations

import datetime as dt
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest

from app.memory import db
from app.memory import memory_manager as mm


class FakeJsonb:
    def __init__(self, value: object) -> None:
        self.value = value


def _install_fake_psycopg(monkeypatch: pytest.MonkeyPatch, connect: object | None = None) -> None:
    psycopg_module = ModuleType("psycopg")
    if connect is not None:
        psycopg_module.connect = connect  # type: ignore[attr-defined]

    types_module = ModuleType("psycopg.types")
    json_module = ModuleType("psycopg.types.json")
    json_module.Jsonb = FakeJsonb  # type: ignore[attr-defined]
    types_module.json = json_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "psycopg", psycopg_module)
    monkeypatch.setitem(sys.modules, "psycopg.types", types_module)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", json_module)


def test_summarize_trace_includes_sorted_keys_and_failure_reason():
    summary = mm.summarize_trace({"b": 1, "a": 2, "failure_reason": "halted"})

    assert "a, b, failure_reason" in summary
    assert "Failure: halted" in summary


def test_embed_text_is_deterministic_and_normalized():
    left = mm.embed_text("same input", dim=8)
    right = mm.embed_text("same input", dim=8)

    assert left == right
    assert len(left) == 8
    assert pytest.approx(sum(value * value for value in left), rel=1e-6) == 1.0


def test_store_trace_uses_jsonb_and_vector_cast(monkeypatch):
    _install_fake_psycopg(monkeypatch)
    trace = {"hypothesis": "alpha", "failure_reason": None}
    calls: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def execute(self, sql: str, params: tuple[object, ...]) -> None:
            calls["sql"] = sql
            calls["params"] = params

    class FakeConnection:
        committed = False

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            self.committed = True

    connection = FakeConnection()

    @contextmanager
    def fake_connection():
        yield connection

    monkeypatch.setattr(mm, "get_connection", fake_connection)

    mm.store_trace(trace)

    params = calls["params"]
    assert "VALUES (%s, %s, %s, %s::vector, %s)" in str(calls["sql"])
    assert isinstance(params[1], FakeJsonb)
    assert params[1].value == trace
    assert isinstance(params[3], str)
    assert str(params[3]).startswith("[")
    assert connection.committed is True


def test_retrieve_similar_formats_rows(monkeypatch):
    now = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    calls: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def execute(self, sql: str, params: tuple[object, ...]) -> None:
            calls["sql"] = sql
            calls["params"] = params

        def fetchall(self) -> list[tuple[object, ...]]:
            return [(1, now, "alpha", {"x": 1}, "summary", None)]

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    monkeypatch.setattr(mm, "get_connection", fake_connection)

    result = mm.retrieve_similar("alpha", limit=3)

    assert "ORDER BY embedding <-> %s::vector" in str(calls["sql"])
    assert calls["params"] == (mm._format_vector(mm.embed_text("alpha")), 3)
    assert result == [
        {
            "id": 1,
            "created_at": now.isoformat(),
            "hypothesis": "alpha",
            "trace": {"x": 1},
            "summary": "summary",
            "failure_reason": None,
        }
    ]


def test_get_database_url_requires_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db._get_database_url()


def test_get_connection_closes_connection(monkeypatch):
    events: list[str] = []

    class FakeConnection:
        def close(self) -> None:
            events.append("closed")

    def connect(url: str) -> FakeConnection:
        events.append(url)
        return FakeConnection()

    _install_fake_psycopg(monkeypatch, connect=connect)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")

    with db.get_connection() as connection:
        assert isinstance(connection, FakeConnection)

    assert events == ["postgresql://example", "closed"]
