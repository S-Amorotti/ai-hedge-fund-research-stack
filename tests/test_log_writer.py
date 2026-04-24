import json
import os
from unittest.mock import patch

import pytest

from app.monitoring.log_writer import DecisionLogger, read_logs, _rotate_if_needed
from app.orchestration.state import GraphState


# ── read_logs ─────────────────────────────────────────────────────────────────


def test_read_logs_missing_file():
    assert read_logs("/nonexistent/path/decisions.log") == []


def test_read_logs_empty_file(tmp_path):
    log = tmp_path / "test.log"
    log.write_text("")
    assert read_logs(str(log)) == []


def test_read_logs_skips_blank_lines(tmp_path):
    log = tmp_path / "test.log"
    log.write_text('{"hypothesis": "h1"}\n\n{"hypothesis": "h2"}\n')
    entries = read_logs(str(log))
    assert len(entries) == 2
    assert entries[0]["hypothesis"] == "h1"
    assert entries[1]["hypothesis"] == "h2"


# ── DecisionLogger ────────────────────────────────────────────────────────────


def test_log_state_writes_jsonl(tmp_path):
    log_path = str(tmp_path / "decisions.log")
    logger = DecisionLogger(log_path=log_path)
    state = GraphState(hypothesis="test hypothesis")

    with patch("app.monitoring.log_writer.store_trace"):
        logger.log_state(state)

    entries = read_logs(log_path)
    assert len(entries) == 1
    assert entries[0]["hypothesis"] == "test hypothesis"
    assert "timestamp" in entries[0]


def test_log_state_accepts_dict(tmp_path):
    log_path = str(tmp_path / "decisions.log")
    logger = DecisionLogger(log_path=log_path)

    with patch("app.monitoring.log_writer.store_trace"):
        logger.log_state({"hypothesis": "dict input"})

    entries = read_logs(log_path)
    assert entries[0]["hypothesis"] == "dict input"


def test_log_state_appends(tmp_path):
    log_path = str(tmp_path / "decisions.log")
    logger = DecisionLogger(log_path=log_path)

    with patch("app.monitoring.log_writer.store_trace"):
        logger.log_state(GraphState(hypothesis="first"))
        logger.log_state(GraphState(hypothesis="second"))

    entries = read_logs(log_path)
    assert len(entries) == 2
    assert entries[0]["hypothesis"] == "first"
    assert entries[1]["hypothesis"] == "second"


# ── log rotation ──────────────────────────────────────────────────────────────


def test_rotation_not_triggered_when_under_limit(tmp_path, monkeypatch):
    log = tmp_path / "decisions.log"
    log.write_text("x" * 100)
    monkeypatch.setenv("MAX_LOG_BYTES", "1000")
    _rotate_if_needed(str(log))
    assert log.exists()
    assert not (tmp_path / "decisions.log.1").exists()


def test_rotation_triggered_when_over_limit(tmp_path, monkeypatch):
    log = tmp_path / "decisions.log"
    log.write_text("x" * 200)
    monkeypatch.setenv("MAX_LOG_BYTES", "100")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "3")
    _rotate_if_needed(str(log))
    assert not log.exists()
    assert (tmp_path / "decisions.log.1").exists()


def test_rotation_shifts_existing_backups(tmp_path, monkeypatch):
    log = tmp_path / "decisions.log"
    log.write_text("x" * 200)
    (tmp_path / "decisions.log.1").write_text("backup1")
    monkeypatch.setenv("MAX_LOG_BYTES", "100")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "3")
    _rotate_if_needed(str(log))
    assert (tmp_path / "decisions.log.2").read_text() == "backup1"


def test_rotation_no_op_if_file_missing(tmp_path):
    _rotate_if_needed(str(tmp_path / "nonexistent.log"))
