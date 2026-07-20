"""Test dell'operazione publish_draft_content (integrazione Laravel/Programmato Backoffice).

Verifica in particolare che la conferma (slot 'confirm') sia rispettata: un valore non
affermativo NON deve mai risultare in una chiamata HTTP di pubblicazione."""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from training_data.operations.publish_draft_content import action_publish_draft_content


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json = MagicMock(return_value=json_data or {})
    return resp


def test_publish_draft_content_no_domain():
    result = action_publish_draft_content("publish_draft_content", slots={"confirm": "si"})
    assert "dominio" in result["response"].lower()


def test_publish_draft_content_confirm_no_skips_api_call():
    with patch("training_data.operations.publish_draft_content.requests.post") as mock_post:
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "no"}
        )

    mock_post.assert_not_called()
    assert "annullat" in result["response"].lower()
    assert result["metadata"]["confirmed"] is False


def test_publish_draft_content_confirm_garbage_skips_api_call():
    with patch("training_data.operations.publish_draft_content.requests.post") as mock_post:
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "forse dopo"}
        )

    mock_post.assert_not_called()
    assert "annullat" in result["response"].lower()


def test_publish_draft_content_confirm_si_calls_api():
    mock_resp = _mock_response(200, {"status": "ok", "domain": "pippo", "count": 2})
    with patch("training_data.operations.publish_draft_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.publish_draft_content.requests.post", return_value=mock_resp) as mock_post:
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "sì"}
        )

    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"] == {"domain": "pippo"}
    assert "2 contenuti in bozza" in result["response"]
    assert result["metadata"]["confirmed"] is True


def test_publish_draft_content_missing_token():
    with patch("training_data.operations.publish_draft_content.config.BACKEND_API_TOKEN", ""):
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "ok"}
        )
    assert "configurata" in result["response"].lower()
    assert result["metadata"]["error"] == "missing_token"


def test_publish_draft_content_zero_count():
    mock_resp = _mock_response(200, {"status": "ok", "domain": "pippo", "count": 0})
    with patch("training_data.operations.publish_draft_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.publish_draft_content.requests.post", return_value=mock_resp):
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "confermo"}
        )

    assert "non ho trovato" in result["response"].lower()


def test_publish_draft_content_network_error():
    with patch("training_data.operations.publish_draft_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.publish_draft_content.requests.post", side_effect=requests.RequestException("boom")):
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "va bene"}
        )

    assert "non riesco a raggiungere" in result["response"].lower()


def test_publish_draft_content_unauthorized():
    mock_resp = _mock_response(401)
    with patch("training_data.operations.publish_draft_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.publish_draft_content.requests.post", return_value=mock_resp):
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "certo"}
        )

    assert "autorizzato" in result["response"].lower()


def test_publish_draft_content_http_error():
    mock_resp = _mock_response(500)
    with patch("training_data.operations.publish_draft_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.publish_draft_content.requests.post", return_value=mock_resp):
        result = action_publish_draft_content(
            "publish_draft_content", slots={"domain": "pippo", "confirm": "procedi"}
        )

    assert result["metadata"]["error"] == "http_500"
