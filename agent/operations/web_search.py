"""Operation per l'intent web_search."""
from ddgs import DDGS

_MAX_BODY_LENGTH = 200


def action_web_search(intent_name: str, slots: dict = None) -> dict:
    """
    Esegue una ricerca web tramite DuckDuckGo.

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili, si aspetta la chiave 'query' con il termine di ricerca

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    query = slots.get("query") or slots.get("QUERY") or slots.get("search_query")

    if not query:
        return {
            "response": "Cosa vuoi che cerchi? Dimmi l'argomento della ricerca.",
            "slots": {},
            "metadata": {"operation": "web_search", "query": None}
        }

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return {
            "response": f"Mi dispiace, non riesco a effettuare la ricerca in questo momento. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "web_search", "query": query, "error": str(e)}
        }

    if not results:
        return {
            "response": f"Non ho trovato risultati per '{query}'. Prova con termini diversi.",
            "slots": {},
            "metadata": {"operation": "web_search", "query": query, "results": []}
        }

    lines = [f"Ecco cosa ho trovato per '{query}':\n"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "")
        body = result.get("body", "")
        href = result.get("href", "")
        lines.append(f"{i}. **{title}**")
        if body:
            lines.append(f"   {body[:_MAX_BODY_LENGTH]}{'...' if len(body) > _MAX_BODY_LENGTH else ''}")
        if href:
            lines.append(f"   🔗 {href}")
        lines.append("")

    response = "\n".join(lines).strip()

    return {
        "response": response,
        "slots": {},
        "metadata": {"operation": "web_search", "query": query, "results": results}
    }
