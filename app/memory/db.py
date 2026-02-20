from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ai_hedge_fund")


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()
