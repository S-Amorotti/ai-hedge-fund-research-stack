from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Copy .env.example to .env and fill in your connection string."
        )
    return url


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(_get_database_url())
    try:
        yield conn
    finally:
        conn.close()
