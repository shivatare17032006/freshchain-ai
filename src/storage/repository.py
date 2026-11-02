"""SQLite persistence: settings, alert acknowledgements, scenarios, chat sessions, run log.
All SQL uses bound parameters."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alert_ack (alert_id TEXT PRIMARY KEY, acknowledged_at TEXT NOT NULL, note TEXT);
CREATE TABLE IF NOT EXISTS scenarios (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at TEXT, params TEXT, results TEXT);
CREATE TABLE IF NOT EXISTS chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, created_at TEXT, role TEXT, content TEXT);
CREATE TABLE IF NOT EXISTS run_log (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT, status TEXT, message TEXT);
"""


class Repository:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.DB_PATH)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:") if str(self.path) == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = self._mem or sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            if self._mem is None:
                conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    # settings
    def get_setting(self, key, default=None):
        with self._conn() as c:
            r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_setting(self, key, value) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value)))

    # alerts
    def acknowledge(self, alert_ids, note: str = "") -> None:
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO alert_ack(alert_id, acknowledged_at, note) VALUES(?,?,?)",
                          [(a, self._now(), note) for a in alert_ids])

    def acknowledged_ids(self) -> set:
        with self._conn() as c:
            return {r[0] for r in c.execute("SELECT alert_id FROM alert_ack")}

    # scenarios
    def save_scenario(self, name: str, params: dict, results: dict) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO scenarios(name,created_at,params,results) VALUES(?,?,?,?)",
                            (name, self._now(), json.dumps(params, default=str), json.dumps(results, default=str)))
            return int(cur.lastrowid)

    def list_scenarios(self) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT id,name,created_at,params,results FROM scenarios ORDER BY id DESC").fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2], "params": json.loads(r[3]), "results": json.loads(r[4])} for r in rows]

    # chat
    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO chat_messages(session_id,created_at,role,content) VALUES(?,?,?,?)",
                      (session_id, self._now(), role, content))

    def get_messages(self, session_id: str) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT role,content,created_at FROM chat_messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def clear_session(self, session_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))

    # run log
    def log_run(self, kind: str, status: str, message: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO run_log(created_at,kind,status,message) VALUES(?,?,?,?)", (self._now(), kind, status, message[:2000]))

    def recent_runs(self, limit: int = 20) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT created_at,kind,status,message FROM run_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"created_at": r[0], "kind": r[1], "status": r[2], "message": r[3]} for r in rows]
