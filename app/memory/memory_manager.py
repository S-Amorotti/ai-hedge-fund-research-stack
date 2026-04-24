from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .db import get_connection


def summarize_trace(trace: dict[str, Any]) -> str:
    """Summarize a trace deterministically before embedding."""

    keys = ", ".join(sorted(trace.keys()))
    failure = trace.get("failure_reason")
    return f"Trace summary with keys: {keys}. Failure: {failure}"


def embed_text(text: str, dim: int = 768) -> list[float]:
    """Deterministic embedding using hashing.

    This avoids external services while remaining reproducible and auditable.
    """

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.normal(0.0, 1.0, size=dim)
    vec = vec / np.linalg.norm(vec)
    return vec.astype(float).tolist()


def _format_vector(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in values) + "]"


def store_trace(trace: dict[str, Any]) -> None:
    from psycopg.types.json import Jsonb

    summary = summarize_trace(trace)
    embedding = embed_text(summary)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO decision_traces (hypothesis, trace, summary, embedding, failure_reason)
                VALUES (%s, %s, %s, %s::vector, %s)
                """,
                (
                    trace.get("hypothesis"),
                    Jsonb(trace),
                    summary,
                    _format_vector(embedding),
                    trace.get("failure_reason"),
                ),
            )
        conn.commit()


def retrieve_similar(summary_query: str, limit: int = 5) -> list[dict[str, Any]]:
    embedding = embed_text(summary_query)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, hypothesis, trace, summary, failure_reason
                FROM decision_traces
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """,
                (_format_vector(embedding), limit),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "created_at": row[1].isoformat(),
            "hypothesis": row[2],
            "trace": row[3],
            "summary": row[4],
            "failure_reason": row[5],
        }
        for row in rows
    ]
