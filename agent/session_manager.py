import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.entity_manager import EntityManager
from agent.persistence import SessionPersistence
from config import BASE_DIR


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

    def _format_log_entry(self, role: str, content: str, intent: str | None) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        role_label = role.upper()
        intent_label = f" intent={intent}" if intent else ""
        return f"[{timestamp}] {role_label}{intent_label}: {content}\n"

    def _append_log_entry(self, role: str, content: str, intent: str | None):
        if not self.log_path:
            return

        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(self._format_log_entry(role, content, intent))
        except OSError:
            pass

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit:
            return self.history[-limit:]
        return self.history

    def clear_history(self):
        self.history = []
        self.updated_at = datetime.now()
        if self.persistence:
            self.persistence.save_session(self)

    def update_context(self, key: str, value: Any):
        self.context[key] = value
        self.updated_at = datetime.now()
        if self.persistence:
            self.persistence.save_session(self)

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)


class SessionManager:
    _instance = None

    def __new__(cls, entity_manager: EntityManager | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
            cls._instance._max_sessions = 1000
            cls._instance._session_timeout = 3600
            cls._instance._entity_manager = entity_manager or EntityManager()
            
            # Persistence
            db_path = os.path.join(BASE_DIR, ".cognitor", "sessions.db")
            cls._instance.persistence = SessionPersistence(db_path)
            
        return cls._instance

    @property
    def entity_manager(self) -> EntityManager:
        return self._entity_manager

    def _get_session_log_path(self, session_id: str) -> str:
        sessions_dir = os.path.join(BASE_DIR, 'sessions')
        os.makedirs(sessions_dir, exist_ok=True)
        return os.path.join(sessions_dir, f"{session_id}.txt")

    def create_session(self, user_id: str | None = None, metadata: dict | None = None) -> str:
        session_id = str(uuid.uuid4())
        log_path = self._get_session_log_path(session_id)

        session = ConversationSession(
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {'user_id': user_id},
            persistence=self.persistence,
            log_path=log_path
        )
        
        self._sessions[session_id] = session
        self.persistence.save_session(session)
        self._cleanup_old_sessions()
        
        return session_id

    def get_session(self, session_id: str) -> ConversationSession | None:
        # Prima prova in memoria
        session = self._sessions.get(session_id)
        if session:
            if self._is_session_valid(session):
                return session
            else:
                del self._sessions[session_id]
                self.persistence.delete_session(session_id)
                return None

        # Poi prova da DB
        session_data = self.persistence.load_session(session_id)
        if session_data:
            session = ConversationSession(
                persistence=self.persistence,
                **session_data
            )
            session.log_path = self._get_session_log_path(session_id)
            if self._is_session_valid(session):
                self._sessions[session_id] = session
                return session
            else:
                self.persistence.delete_session(session_id)
        
        return None

    def delete_session(self, session_id: str) -> bool:
        deleted = False
        if session_id in self._sessions:
            del self._sessions[session_id]
            deleted = True
        
        self.persistence.delete_session(session_id)
        return deleted

    def _is_session_valid(self, session: ConversationSession) -> bool:
        elapsed = (datetime.now() - session.updated_at).total_seconds()
        return elapsed < self._session_timeout

    def _cleanup_old_sessions(self):
        # Purga anche il DB dalle sessioni scadute per timeout, altrimenti quelle
        # abbandonate (mai più riaccedute via get_session) restano in sqlite per sempre.
        self.persistence.cleanup_old_sessions(self._session_timeout)

        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].updated_at
            )
            to_remove = len(self._sessions) - self._max_sessions + 100
            for session_id, _ in sorted_sessions[:to_remove]:
                del self._sessions[session_id]

    def get_active_sessions(self) -> list[str]:
        # Qui potremmo voler interrogare anche il DB per sessioni non in memoria ma valide
        return list(self._sessions.keys())

    def set_session_timeout(self, seconds: int):
        self._session_timeout = seconds

    def set_max_sessions(self, max_count: int):
        self._max_sessions = max_count
