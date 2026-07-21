import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Any

class SessionPersistence:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    updated_at TEXT,
                    history TEXT,
                    context TEXT,
                    metadata TEXT,
                    waiting_for_slot TEXT,
                    agent_mode TEXT
                )
            """)

    def save_session(self, session):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, created_at, updated_at, history, context, metadata, waiting_for_slot, agent_mode)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    json.dumps(session.history),
                    json.dumps(session.context),
                    json.dumps(session.metadata),
                    json.dumps(session.waiting_for_slot) if session.waiting_for_slot else None,
                    session.agent_mode
                ))
        except sqlite3.OperationalError as e:
            print(f"WARNING Persistence: impossibile salvare la sessione {session.session_id}: {e}")

    def load_session(self, session_id: str) -> Optional[dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        "session_id": row[0],
                        "created_at": datetime.fromisoformat(row[1]),
                        "updated_at": datetime.fromisoformat(row[2]),
                        "history": json.loads(row[3]),
                        "context": json.loads(row[4]),
                        "metadata": json.loads(row[5]),
                        "waiting_for_slot": json.loads(row[6]) if row[6] else None,
                        "agent_mode": row[7]
                    }
        except sqlite3.OperationalError as e:
            print(f"WARNING Persistence: impossibile caricare la sessione {session_id}: {e}")
        return None

    def delete_session(self, session_id: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        except sqlite3.OperationalError as e:
            print(f"WARNING Persistence: impossibile eliminare la sessione {session_id}: {e}")

    def cleanup_old_sessions(self, timeout_seconds: int) -> int:
        """
        Elimina dal DB le sessioni la cui updated_at è più vecchia di timeout_seconds.

        Args:
            timeout_seconds: soglia in secondi oltre la quale una sessione è considerata scaduta

        Returns:
            Numero di righe eliminate (0 in caso di errore di persistenza)
        """
        cutoff = datetime.now().timestamp() - timeout_seconds
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE updated_at < ?", (cutoff_iso,)
                )
                return cursor.rowcount
        except sqlite3.OperationalError as e:
            print(f"WARNING Persistence: impossibile eseguire cleanup delle sessioni scadute: {e}")
            return 0
