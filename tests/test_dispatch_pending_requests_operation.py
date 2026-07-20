"""Test dell'operazione dispatch_pending_requests (integrazione Laravel/Programmato Backoffice)."""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from training_data.operations.dispatch_pending_requests import action_dispatch_pending_requests


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json = MagicMock(return_value=json_data or {})
    return resp


def test_dispatch_pending_requests_missing_token():
    with patch("training_data.operations.dispatch_pending_requests.config.BACKEND_API_TOKEN", ""):
        result = action_dispatch_pending_requests("dispatch_pending_requests", slots={})
    assert "configurato" in result["response"].lower()
    assert result["metadata"]["error"] == "missing_token"


def test_dispatch_pending_requests_success():
    mock_resp = _mock_response(200, {"status": "ok", "count": 4})
    with patch("training_data.operations.dispatch_pending_requests.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.dispatch_pending_requests.requests.post", return_value=mock_resp) as mock_post:
        result = action_dispatch_pending_requests("dispatch_pending_requests", slots={})

    assert "4 richieste in sospeso" in result["response"]
    assert result["metadata"]["count"] == 4
    mock_post.assert_called_once()


def test_dispatch_pending_requests_zero_count():
    mock_resp = _mock_response(200, {"status": "ok", "count": 0})
    with patch("training_data.operations.dispatch_pending_requests.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.dispatch_pending_requests.requests.post", return_value=mock_resp):
        result = action_dispatch_pending_requests("dispatch_pending_requests", slots={})

    assert "non c'erano richieste" in result["response"].lower()


def test_dispatch_pending_requests_network_error():
    with patch("training_data.operations.dispatch_pending_requests.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.dispatch_pending_requests.requests.post", side_effect=requests.RequestException("boom")):
        result = action_dispatch_pending_requests("dispatch_pending_requests", slots={})

    assert "non riesco a raggiungere" in result["response"].lower()
    assert "error" in result["metadata"]


def test_dispatch_pending_requests_unauthorized():
    mock_resp = _mock_response(401)
    with patch("training_data.operations.dispatch_pending_requests.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.dispatch_pending_requests.requests.post", return_value=mock_resp):
        result = action_dispatch_pending_requests("dispatch_pending_requests", slots={})

    assert "autorizzato" in result["response"].lower()
    assert result["metadata"]["error"] == "unauthorized"


def test_dispatch_pending_requests_http_error():
    mock_resp = _mock_response(500)
    with patch("training_data.operations.dispatch_pending_requests.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.dispatch_pending_requests.requests.post", return_value=mock_resp):
        result = action_dispatch_pending_requests("dispatch_pending_requests", slots={})

    assert result["metadata"]["error"] == "http_500"
