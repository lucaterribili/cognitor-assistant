"""Test del contratto chat v1 (api/chat_v1.py): token OAuth2 client_credentials + chat."""
import base64
import os
from types import SimpleNamespace

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("AUTH_ADMIN_PASSWORD", "admin-pass")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import chat_v1
from api.dependencies import get_agent


class FakeSessionManager:
    def __init__(self):
        self.sessions = {}

    def get_session(self, sid):
        return self.sessions.get(sid)

    def create_session(self, user_id=None, metadata=None, session_id=None):
        self.sessions[session_id] = SimpleNamespace(user=user_id)
        return session_id

    def delete_session(self, sid):
        return self.sessions.pop(sid, None) is not None


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("COGNITOR_CLIENTS", "site-a:sec:ret,site-b:other")
    agent = SimpleNamespace(session_manager=FakeSessionManager())
    seen = []

    class FakeTurn:
        def __init__(self, a):
            pass

        def process(self, text, session):
            seen.append((text, session))
            return SimpleNamespace(response=f"echo:{text}", options=None, wait_for_slot=None)

    monkeypatch.setattr(chat_v1, "TurnProcessor", FakeTurn)
    app = FastAPI()
    app.include_router(chat_v1.oauth_router, prefix="/api/oauth2")
    app.include_router(chat_v1.chat_router, prefix="/api/v1")
    app.dependency_overrides[get_agent] = lambda: agent
    return SimpleNamespace(client=TestClient(app), agent=agent, seen=seen)


def basic(cid, secret):
    return {"Authorization": "Basic " + base64.b64encode(f"{cid}:{secret}".encode()).decode()}


def get_token(client, cid="site-a", secret="sec:ret"):
    r = client.post("/api/oauth2/token", headers=basic(cid, secret),
                    data={"grant_type": "client_credentials", "scope": "CHAT"})
    assert r.status_code == 200, r.text
    return r.json()


def bearer(tok):
    return {"Authorization": f"Bearer {tok['access_token']}"}


def test_token_shape_matches_spring(env):
    body = get_token(env.client)
    assert body["token_type"] == "Bearer"
    assert body["expires_in"] > 0 and body["access_token"]


@pytest.mark.parametrize("cid,secret", [("site-a", "wrong"), ("nobody", "x"), ("site-a", "password")])
def test_token_rejects_bad_credentials(env, cid, secret):
    r = env.client.post("/api/oauth2/token", headers=basic(cid, secret),
                        data={"grant_type": "client_credentials", "scope": "CHAT"})
    assert r.status_code == 401


def test_token_requires_client_credentials_grant_and_known_scope(env):
    h = basic("site-a", "sec:ret")
    assert env.client.post("/api/oauth2/token", headers=h, data={"grant_type": "password"}).status_code == 400
    assert env.client.post("/api/oauth2/token", headers=h,
                           data={"grant_type": "client_credentials", "scope": "ADMIN"}).status_code == 400
    assert env.client.post("/api/oauth2/token", data={"grant_type": "client_credentials"}).status_code == 401


def test_token_accepts_credentials_in_body(env):
    r = env.client.post("/api/oauth2/token", data={
        "grant_type": "client_credentials", "client_id": "site-b", "client_secret": "other"})
    assert r.status_code == 200


def test_default_client_is_admin_user(env, monkeypatch):
    monkeypatch.delenv("COGNITOR_CLIENTS")
    get_token(env.client, chat_v1._ADMIN_USERNAME, chat_v1._ADMIN_PASSWORD)


def test_chat_requires_bearer(env):
    assert env.client.post("/api/v1/chat", json={"session_id": "s", "text": "ciao"}).status_code == 401


def test_chat_contract_and_aliases(env):
    h = bearer(get_token(env.client))
    for payload in ({"session_id": "s1", "text": "ciao"},
                    {"sessionId": "s1", "text": "ciao"},
                    {"sender": "s1", "message": "ciao"}):
        r = env.client.post("/api/v1/chat", json=payload, headers=h)
        assert r.status_code == 200, r.text
        assert r.json() == [{"session_id": "s1", "text": "echo:ciao", "link": None,
                             "options": None, "waiting_for_slot": False}]


def test_chat_generates_session_id_when_missing_and_rejects_empty_text(env):
    h = bearer(get_token(env.client))
    r = env.client.post("/api/v1/chat", json={"text": "ciao"}, headers=h)
    assert r.json()[0]["session_id"]
    assert env.client.post("/api/v1/chat", json={"text": "  "}, headers=h).status_code == 400


def test_sessions_are_isolated_per_client(env):
    a, b = bearer(get_token(env.client)), bearer(get_token(env.client, "site-b", "other"))
    env.client.post("/api/v1/chat", json={"session_id": "user", "text": "x"}, headers=a)
    env.client.post("/api/v1/chat", json={"session_id": "user", "text": "y"}, headers=b)
    assert set(env.agent.session_manager.sessions) == {"site-a__user", "site-b__user"}


def test_unsafe_session_id_never_reaches_filesystem(env):
    h = bearer(get_token(env.client))
    evil = "../../etc/passwd"
    r = env.client.post("/api/v1/chat", json={"session_id": evil, "text": "x"}, headers=h)
    assert r.json()[0]["session_id"] == evil  # il client rivede il proprio id
    (internal,) = env.agent.session_manager.sessions
    assert "/" not in internal and ".." not in internal


def test_reset_drops_session(env):
    h = bearer(get_token(env.client))
    env.client.post("/api/v1/chat", json={"session_id": "s", "text": "x"}, headers=h)
    assert env.client.post("/api/v1/chat/reset", json={"session_id": "s"}, headers=h).status_code == 200
    assert not env.agent.session_manager.sessions
