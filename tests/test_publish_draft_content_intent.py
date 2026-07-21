"""Test end-to-end (via Agent.get_response) del flusso di conferma di publish_draft_content.

Richiede che `.cognitor/rules.yaml`/`responses.yaml` siano stati rigenerati con
`python -m pipeline.merge_data` (o l'intera pipeline) dopo l'aggiunta dei file in
training_data/{intents,rules,responses}/laravel_publish_draft_content.yaml."""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import Agent


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json = MagicMock(return_value=json_data or {})
    return resp


def test_publish_draft_content_asks_domain_then_confirm():
    agent = Agent()
    agent.load_models()
    agent.load_knowledge()

    response_text, wait_slot, bot_slots = agent.get_response(
        "publish_draft_content", {}, []
    )
    assert wait_slot == "domain"

    response_text, wait_slot, bot_slots = agent.get_response(
        "publish_draft_content", {"domain": "pippo"}, []
    )
    assert wait_slot == "confirm"
    assert "confer" in response_text.lower() or "sì" in response_text.lower()


def test_publish_draft_content_confirm_no_does_not_call_api():
    agent = Agent()
    agent.load_models()
    agent.load_knowledge()

    with patch("training_data.operations.publish_draft_content.requests.post") as mock_post:
        response_text, wait_slot, bot_slots = agent.get_response(
            "publish_draft_content", {"domain": "pippo", "confirm": "no"}, []
        )

    mock_post.assert_not_called()
    assert wait_slot is None
    assert "annullat" in response_text.lower()


def test_publish_draft_content_confirm_si_calls_api():
    agent = Agent()
    agent.load_models()
    agent.load_knowledge()

    mock_resp = _mock_response(200, {"status": "ok", "domain": "pippo", "count": 1})
    with patch("training_data.operations.publish_draft_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.publish_draft_content.requests.post", return_value=mock_resp) as mock_post:
        response_text, wait_slot, bot_slots = agent.get_response(
            "publish_draft_content", {"domain": "pippo", "confirm": "sì"}, []
        )

    mock_post.assert_called_once()
    assert wait_slot is None
    assert "pubblicato" in response_text.lower()
