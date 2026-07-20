"""Test dell'operazione optimize_content (integrazione Laravel/Programmato Backoffice)."""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from training_data.operations.optimize_content import action_optimize_content


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json = MagicMock(return_value=json_data or {})
    return resp


def test_optimize_content_no_domain():
    result = action_optimize_content("optimize_content", slots={})
    assert "dominio" in result["response"].lower()
    assert result["metadata"]["domain"] is None


def test_optimize_content_missing_token():
    with patch("training_data.operations.optimize_content.config.BACKEND_API_TOKEN", ""):
        result = action_optimize_content("optimize_content", slots={"domain": "pippo"})
    assert "configurata" in result["response"].lower()
    assert result["metadata"]["error"] == "missing_token"


def test_optimize_content_success():
    mock_resp = _mock_response(200, {"status": "queued", "domain": "pippo", "count": 3})
    with patch("training_data.operations.optimize_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.optimize_content.requests.post", return_value=mock_resp) as mock_post:
        result = action_optimize_content("optimize_content", slots={"domain": "pippo"})

    assert "3 contenuti in bozza" in result["response"]
    assert "pippo" in result["response"]
    assert result["metadata"]["count"] == 3
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"] == {"domain": "pippo"}


def test_optimize_content_zero_count():
    mock_resp = _mock_response(200, {"status": "queued", "domain": "pippo", "count": 0})
    with patch("training_data.operations.optimize_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.optimize_content.requests.post", return_value=mock_resp):
        result = action_optimize_content("optimize_content", slots={"domain": "pippo"})

    assert "non ho trovato" in result["response"].lower()
    assert result["metadata"]["count"] == 0


def test_optimize_content_network_error():
    with patch("training_data.operations.optimize_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.optimize_content.requests.post", side_effect=requests.RequestException("boom")):
        result = action_optimize_content("optimize_content", slots={"domain": "pippo"})

    assert "non riesco a raggiungere" in result["response"].lower()
    assert "error" in result["metadata"]


def test_optimize_content_unauthorized():
    mock_resp = _mock_response(401)
    with patch("training_data.operations.optimize_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.optimize_content.requests.post", return_value=mock_resp):
        result = action_optimize_content("optimize_content", slots={"domain": "pippo"})

    assert "autorizzato" in result["response"].lower()
    assert result["metadata"]["error"] == "unauthorized"


def test_optimize_content_http_error():
    mock_resp = _mock_response(500)
    with patch("training_data.operations.optimize_content.config.BACKEND_API_TOKEN", "fake-token"), \
         patch("training_data.operations.optimize_content.requests.post", return_value=mock_resp):
        result = action_optimize_content("optimize_content", slots={"domain": "pippo"})

    assert result["metadata"]["error"] == "http_500"
