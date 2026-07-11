"""Operation per l'intent proposal_status — verifica lo stato delle Proposal generate via chatbot
(articoli/tutorial creati con create_content) nel backoffice Laravel (Programmato)."""
import requests

import config

_MAX_RESULTS = 5

_STATUS_LABELS = {
    "pending": "in attesa di essere presa in carico",
    "running": "in elaborazione",
    "completed": "completata",
    "error": "in errore",
}


def _status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


def action_proposal_status(intent_name: str, slots: dict = None) -> dict:
    """
    Verifica lo stato delle proposal generate dal bot (task generate-article/generate-tutorial)
    tramite l'endpoint scoped GET /api/chatbot/proposals?query=... (richiede un token con
    ability "proposals:read", vedi comando artisan `chatbot:cognitor-token`).

    Args:
        intent_name: Nome dell'intent
        slots: Slot disponibili, opzionalmente la chiave 'query' per filtrare per argomento

    Returns:
        dict con la risposta
    """
    slots = slots or {}
    query = slots.get("query") or slots.get("PROPOSAL_QUERY")

    if not config.BACKEND_API_TOKEN:
        return {
            "response": "Il controllo delle richieste non è ancora configurato (manca il token di accesso al backoffice).",
            "slots": {},
            "metadata": {"operation": "proposal_status", "query": query, "error": "missing_token"},
        }

    url = f"{config.BACKEND_API_BASE_URL.rstrip('/')}/chatbot/proposals"
    params = {}
    if query:
        params["query"] = query

    try:
        resp = requests.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {config.BACKEND_API_TOKEN}",
                "Accept": "application/json",
            },
            timeout=config.BACKEND_API_TIMEOUT,
        )
    except requests.RequestException as e:
        return {
            "response": "Non riesco a raggiungere il backoffice in questo momento. Riprova più tardi.",
            "slots": {},
            "metadata": {"operation": "proposal_status", "query": query, "error": str(e)},
        }

    if resp.status_code == 401:
        return {
            "response": "Non sono autorizzato a controllare le richieste (token non valido o scaduto).",
            "slots": {},
            "metadata": {"operation": "proposal_status", "query": query, "error": "unauthorized"},
        }

    if not resp.ok:
        return {
            "response": "Il backoffice ha risposto con un errore, non sono riuscito a recuperare lo stato.",
            "slots": {},
            "metadata": {"operation": "proposal_status", "query": query, "error": f"http_{resp.status_code}"},
        }

    try:
        payload = resp.json()
    except ValueError:
        return {
            "response": "Ho ricevuto una risposta inattesa dal backoffice.",
            "slots": {},
            "metadata": {"operation": "proposal_status", "query": query, "error": "invalid_json"},
        }

    proposals = payload if isinstance(payload, list) else payload.get("data", [])

    if not proposals:
        message = (
            f"Non ho trovato richieste che corrispondono a '{query}'."
            if query
            else "Non hai richieste di generazione contenuti in corso."
        )
        return {
            "response": message,
            "slots": {},
            "metadata": {"operation": "proposal_status", "query": query, "results": []},
        }

    lines = ["Ecco lo stato delle tue richieste:\n"]
    for i, proposal in enumerate(proposals[:_MAX_RESULTS], 1):
        titles = proposal.get("titles") or []
        subject = ", ".join(titles) if titles else (proposal.get("tutorial") or proposal.get("category") or proposal.get("name"))
        status = _status_label(proposal.get("status", ""))
        lines.append(f"{i}. {subject} — {status}")

    return {
        "response": "\n".join(lines),
        "slots": {},
        "metadata": {"operation": "proposal_status", "query": query, "results": proposals},
    }
