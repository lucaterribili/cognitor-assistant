"""
Sessioni di conversazione: stato in memoria + persistenza su SQLite
(.cognitor/sessions.db) + log testuale per sessione (sessions/<id>.txt).
"""
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from config import BASE_DIR


class SessionPersistence:
    """Salvataggio/caricamento delle sessioni su SQLite. Gli errori di I/O non
    devono mai bloccare la conversazione: vengono solo segnalati a console."""

    def __init__(self, db_path: str):
        self.db_path = db_path
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
                row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
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
        """Elimina le sessioni con updated_at più vecchia di timeout_seconds.
        Ritorna il numero di righe eliminate (0 in caso di errore)."""
        cutoff_iso = datetime.fromtimestamp(datetime.now().timestamp() - timeout_seconds).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff_iso,)).rowcount
        except sqlite3.OperationalError as e:
            print(f"WARNING Persistence: impossibile eseguire cleanup delle sessioni scadute: {e}")
            return 0


@dataclass
class ConversationSession:
    session_id: str
    created_at: datetime
    updated_at: datetime
    history: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    waiting_for_slot: dict | None = None
    agent_mode: str = "predictable"
    persistence: Any = None
    log_path: str | None = None

    def add_message(self, role: str, content: str, intent: str | None = None, entities: list | None = None):
        self.history.append({
            'role': role,
            'content': content,
            'intent': intent,
            'entities': entities or [],
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
        self._append_log_entry(role, content, intent)
        if self.persistence:
            self.persistence.save_session(self)

    def _append_log_entry(self, role: str, content: str, intent: str | None):
        if not self.log_path:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        intent_label = f" intent={intent}" if intent else ""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {role.upper()}{intent_label}: {content}\n")
        except OSError:
            pass

    def update_context(self, key: str, value: Any):
        self.context[key] = value
        self.updated_at = datetime.now()
        if self.persistence:
            self.persistence.save_session(self)


class SessionManager:
    """Singleton: tutte le istanze condividono le stesse sessioni in memoria."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
            cls._instance._max_sessions = 1000
            cls._instance._session_timeout = 3600
            cls._instance.persistence = SessionPersistence(os.path.join(BASE_DIR, ".cognitor", "sessions.db"))
        return cls._instance

    @staticmethod
    def _get_session_log_path(session_id: str) -> str:
        sessions_dir = os.path.join(BASE_DIR, 'sessions')
        os.makedirs(sessions_dir, exist_ok=True)
        return os.path.join(sessions_dir, f"{session_id}.txt")

    def create_session(self, user_id: str | None = None, metadata: dict | None = None, session_id: str | None = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {'user_id': user_id},
            persistence=self.persistence,
            log_path=self._get_session_log_path(session_id)
        )
        self._sessions[session_id] = session
        self.persistence.save_session(session)
        self._cleanup_old_sessions()
        return session_id

    def get_session(self, session_id: str) -> ConversationSession | None:
        # Prima in memoria, poi da DB
        session = self._sessions.get(session_id)
        if session is None:
            session_data = self.persistence.load_session(session_id)
            if not session_data:
                return None
            session = ConversationSession(persistence=self.persistence, **session_data)
            session.log_path = self._get_session_log_path(session_id)

        if not self._is_session_valid(session):
            self._sessions.pop(session_id, None)
            self.persistence.delete_session(session_id)
            return None

        self._sessions[session_id] = session
        return session

    def delete_session(self, session_id: str) -> bool:
        deleted = self._sessions.pop(session_id, None) is not None
        self.persistence.delete_session(session_id)
        return deleted

    def get_active_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def _is_session_valid(self, session: ConversationSession) -> bool:
        return (datetime.now() - session.updated_at).total_seconds() < self._session_timeout

    def _cleanup_old_sessions(self):
        # Purga anche il DB dalle sessioni scadute per timeout, altrimenti quelle
        # abbandonate (mai più riaccedute via get_session) restano in sqlite per sempre.
        self.persistence.cleanup_old_sessions(self._session_timeout)

        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1].updated_at)
            to_remove = len(self._sessions) - self._max_sessions + 100
            for session_id, _ in sorted_sessions[:to_remove]:
                del self._sessions[session_id]
