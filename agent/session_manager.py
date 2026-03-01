import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.entity_manager import EntityManager


@dataclass
class ConversationSession:
    session_id: str
    created_at: datetime
    updated_at: datetime
    history: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    waiting_for_slot: dict | None = None

    def add_message(self, role: str, content: str, intent: str | None = None, entities: list | None = None):
        self.history.append({
            'role': role,
            'content': content,
            'intent': intent,
            'entities': entities or [],
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit:
            return self.history[-limit:]
        return self.history

    def clear_history(self):
        self.history = []
        self.updated_at = datetime.now()

    def update_context(self, key: str, value: Any):
        self.context[key] = value
        self.updated_at = datetime.now()

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
        return cls._instance

    @property
    def entity_manager(self) -> EntityManager:
        return self._entity_manager

    def create_session(self, user_id: str | None = None, metadata: dict | None = None) -> str:
        session_id = str(uuid.uuid4())
        
        session = ConversationSession(
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {'user_id': user_id}
        )
        
        self._sessions[session_id] = session
        self._cleanup_old_sessions()
        
        return session_id

    def get_session(self, session_id: str) -> ConversationSession | None:
        session = self._sessions.get(session_id)
        if session and self._is_session_valid(session):
            return session
        elif session:
            del self._sessions[session_id]
        return None

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def _is_session_valid(self, session: ConversationSession) -> bool:
        elapsed = (datetime.now() - session.updated_at).total_seconds()
        return elapsed < self._session_timeout

    def _cleanup_old_sessions(self):
        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].updated_at
            )
            to_remove = len(self._sessions) - self._max_sessions + 100
            for session_id, _ in sorted_sessions[:to_remove]:
                del self._sessions[session_id]

    def get_active_sessions(self) -> list[str]:
        self._cleanup_old_sessions()
        return list(self._sessions.keys())

    def set_session_timeout(self, seconds: int):
        self._session_timeout = seconds

    def set_max_sessions(self, max_count: int):
        self._max_sessions = max_count
