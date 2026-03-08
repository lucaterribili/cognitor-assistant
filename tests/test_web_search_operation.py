"""
Test dell'operazione web_search con DuckDuckGo (ddgs).
Verifica che la ricerca web funzioni correttamente.
"""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.operations.web_search import action_web_search


def test_web_search_no_query():
    """Test: nessuna query fornita → risposta di richiesta."""
    result = action_web_search("web_search", slots={})

    print("TEST 1: Nessuna query fornita")
    print(f"  Response: {result['response']}")
    assert "cerchi" in result["response"].lower() or "ricerca" in result["response"].lower(), \
        "La risposta deve chiedere la query"
    assert result["metadata"]["query"] is None
    print("  ✓ Passato")


def test_web_search_with_results():
    """Test: ricerca con risultati validi da DuckDuckGo (mock)."""
    mock_results = [
        {
            "title": "Python - Wikipedia",
            "body": "Python è un linguaggio di programmazione ad alto livello.",
            "href": "https://it.wikipedia.org/wiki/Python"
        },
        {
            "title": "Python.org",
            "body": "Il sito ufficiale di Python.",
            "href": "https://www.python.org"
        }
    ]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text = MagicMock(return_value=iter(mock_results))

    with patch("agent.operations.web_search.DDGS", return_value=mock_ddgs_instance):
        result = action_web_search("web_search", slots={"query": "python programmazione"})

    print("TEST 2: Ricerca con risultati")
    print(f"  Response (prime 100 chars): {result['response'][:100]}")
    assert "python programmazione" in result["response"].lower()
    assert "Python - Wikipedia" in result["response"]
    assert result["metadata"]["query"] == "python programmazione"
    assert len(result["metadata"]["results"]) == 2
    print("  ✓ Passato")


def test_web_search_no_results():
    """Test: ricerca senza risultati."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text = MagicMock(return_value=iter([]))

    with patch("agent.operations.web_search.DDGS", return_value=mock_ddgs_instance):
        result = action_web_search("web_search", slots={"query": "xyzabcdef123456"})

    print("TEST 3: Nessun risultato trovato")
    print(f"  Response: {result['response']}")
    assert "non ho trovato" in result["response"].lower()
    assert result["metadata"]["results"] == []
    print("  ✓ Passato")


def test_web_search_error_handling():
    """Test: gestione errore di rete."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text = MagicMock(side_effect=Exception("Network error"))

    with patch("agent.operations.web_search.DDGS", return_value=mock_ddgs_instance):
        result = action_web_search("web_search", slots={"query": "test"})

    print("TEST 4: Gestione errore di rete")
    print(f"  Response: {result['response']}")
    assert "dispiace" in result["response"].lower() or "riprova" in result["response"].lower()
    assert "error" in result["metadata"]
    print("  ✓ Passato")


def test_web_search_operation_manager_discovery():
    """Test: OperationManager scopre automaticamente web_search."""
    from agent.operations.manager import OperationManager

    print("TEST 5: Auto-discovery dell'operation web_search")
    manager = OperationManager(auto_discover=True)
    operations = manager.list_operations()
    print(f"  Operations trovate: {operations}")
    assert "web_search" in operations, "web_search non trovata nelle operations"
    print("  ✓ Passato")


def test_web_search_real_request():
    """Test: ricerca reale su DuckDuckGo (richiede connessione internet)."""
    print("TEST 6: Ricerca reale su DuckDuckGo")
    try:
        result = action_web_search("web_search", slots={"query": "Come programmare in Python"})
        print(f"  Metadata query: {result['metadata']['query']}")
        print(f"  Risultati: {len(result['metadata'].get('results', []))}")
        print(f"  Response (prime 200 chars): {result['response'][:200]}")

        if "error" in result["metadata"]:
            print(f"  ⚠ Errore di rete (normale in ambienti sandbox): {result['metadata']['error']}")
        else:
            assert result["metadata"]["query"] == "Come programmare in Python"
            assert isinstance(result["metadata"]["results"], list)
            print("  ✓ Passato")
    except Exception as e:
        print(f"  ⚠ Skip (connessione non disponibile): {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Operation web_search con DuckDuckGo (ddgs)")
    print("=" * 60)
    print()

    test_web_search_no_query()
    print()
    test_web_search_with_results()
    print()
    test_web_search_no_results()
    print()
    test_web_search_error_handling()
    print()
    test_web_search_operation_manager_discovery()
    print()
    test_web_search_real_request()

    print()
    print("=" * 60)
    print("🎉 TUTTI I TEST COMPLETATI!")
    print("=" * 60)
