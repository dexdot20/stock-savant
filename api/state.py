"""Shared task state for API (SQLite-backed persistent repository)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core.paths import get_runtime_dir

_DB_DIR = get_runtime_dir() / "instance"
_DB_PATH = _DB_DIR / "api_tasks.sqlite3"
_DB_LOCK = threading.Lock()
_ACTIVE_STATUSES = frozenset({"Pending", "In Progress"})
_TERMINAL_STATUSES = ("Completed", "Failed", "Cancelled")


def _ensure_db() -> None:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_created_ts
            ON tasks(created_ts)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_updated_ts
            ON tasks(updated_ts)
            """
        )
        conn.commit()


# Initialize the database once when the module is loaded; no need to call it per function.
_ensure_db()


def _serialize_payload(payload: Dict[str, Any]) -> str:
    def _default(value: Any) -> Any:
        if hasattr(value, "value"):
            return value.value
        return str(value)

    return json.dumps(payload, ensure_ascii=False, default=_default)


def _task_created_at_ts(task: Dict[str, Any]) -> float:
    ts = task.get("created_ts")
    if isinstance(ts, (int, float)):
        return float(ts)

    created_at = task.get("created_at")
    if isinstance(created_at, str):
        try:
            return datetime.fromisoformat(created_at).timestamp()
        except ValueError:
            pass

    return time.time()


def create_task(task_id: str, payload: Dict[str, Any]) -> None:
    """Create or replace a task payload by id."""
    now = time.time()
    row_payload = dict(payload)
    if "task_id" not in row_payload:
        row_payload["task_id"] = task_id
    if "created_ts" not in row_payload:
        row_payload["created_ts"] = now
    if "created_at" not in row_payload:
        row_payload["created_at"] = datetime.now().isoformat()

    serialized = _serialize_payload(row_payload)
    created_ts = _task_created_at_ts(row_payload)

    with _DB_LOCK:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO tasks(task_id, payload, created_ts, updated_ts)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    payload = excluded.payload,
                    created_ts = excluded.created_ts,
                    updated_ts = excluded.updated_ts
                """,
                (task_id, serialized, created_ts, now),
            )
            conn.commit()


def update_task(task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Patch an existing task and return updated payload, or None if missing."""
    existing = get_task(task_id)
    if not existing:
        return None

    merged = dict(existing)
    merged.update(updates)
    merged["updated_at"] = datetime.now().isoformat()
    create_task(task_id, merged)
    return merged


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute(
                "SELECT payload FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def has_task(task_id: str) -> bool:
    return get_task(task_id) is not None


def reconcile_incomplete_tasks(
    message: str = "Task terminated because the application was restarted.",
) -> int:
    """Mark persisted non-terminal tasks as failed after an application restart."""
    reconciled = 0
    with _DB_LOCK:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute("SELECT task_id, payload FROM tasks").fetchall()
            for task_id, raw_payload in rows:
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("status") not in _ACTIVE_STATUSES:
                    continue

                payload["status"] = "Failed"
                payload["success"] = False
                payload["message"] = message
                payload["updated_at"] = datetime.now().isoformat()
                now = time.time()
                conn.execute(
                    "UPDATE tasks SET payload = ?, updated_ts = ? WHERE task_id = ?",
                    (_serialize_payload(payload), now, task_id),
                )
                reconciled += 1
            conn.commit()
    return reconciled


def purge_expired_tasks(ttl_seconds: int = 24 * 60 * 60) -> int:
    """Delete expired tasks and return number of deleted rows."""
    threshold = time.time() - max(0, int(ttl_seconds))
    with _DB_LOCK:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT task_id, payload, updated_ts FROM tasks WHERE updated_ts < ?",
                (threshold,),
            ).fetchall()

            expired_task_ids = []
            for task_id, raw_payload, _updated_ts in rows:
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    expired_task_ids.append(task_id)
                    continue
                if not isinstance(payload, dict):
                    expired_task_ids.append(task_id)
                    continue
                if payload.get("status") in _TERMINAL_STATUSES:
                    expired_task_ids.append(task_id)

            if not expired_task_ids:
                return 0

            placeholders = ", ".join("?" for _ in expired_task_ids)
            cursor = conn.execute(
                f"DELETE FROM tasks WHERE task_id IN ({placeholders})",
                expired_task_ids,
            )
            conn.commit()
            return int(cursor.rowcount or 0)
