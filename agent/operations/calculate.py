"""Operation per l'intent calculate.

L'intent `calculate` non ha uno slot NER dedicato per l'operatore (solo i numeri
sono taggati come NUMBER negli esempi di training): l'operatore viene invece
riconosciuto con un'estrazione mirata (regex) direttamente dal testo grezzo del
turno (`raw_text`), sia in forma simbolica ("2+2") sia a parole ("5 più 3").
Nessun eval: i due numeri e l'operatore vengono estratti singolarmente da un
pattern fisso e applicati con le funzioni del modulo `operator` — non c'è mai
esecuzione di codice arbitrario.
"""

import operator
import re

_NUMBER = r"-?\d+(?:[.,]\d+)?"

_OPERATOR_FUNCS = {
    "+": operator.add,
    "più": operator.add,
    "piu": operator.add,
    "-": operator.sub,
    "meno": operator.sub,
    "*": operator.mul,
    "x": operator.mul,
    "per": operator.mul,
    "/": operator.truediv,
    "diviso": operator.truediv,
    "fratto": operator.truediv,
}

_OPERATOR_PATTERN = "|".join(
    re.escape(token) for token in sorted(_OPERATOR_FUNCS, key=len, reverse=True)
)

_EXPRESSION_RE = re.compile(
    rf"({_NUMBER})\s*({_OPERATOR_PATTERN})\s*({_NUMBER})",
    re.IGNORECASE,
)


def _parse_number(raw: str) -> int | float:
    value = float(raw.replace(",", "."))
    return int(value) if value.is_integer() else value


def _format_result(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def action_calculate(intent_name: str = None, slots: dict = None, raw_text: str = None) -> dict:
    """
    Estrae un'espressione aritmetica a due operandi dal messaggio dell'utente
    e la calcola.

    Args:
        intent_name: Nome dell'intent che ha attivato l'operation
        slots: Slot disponibili (non usati qui)
        raw_text: Testo grezzo del turno corrente, da cui estrarre l'espressione

    Returns:
        dict con la risposta e eventuali metadati
    """
    match = _EXPRESSION_RE.search(raw_text or "")
    if not match:
        return {
            "response": (
                "Non sono riuscito a capire l'espressione da calcolare. "
                "Prova a scrivermela così: \"quanto fa 5 più 3\" oppure \"5+3\"."
            ),
            "slots": {},
            "metadata": {"operation": "calculate", "error": "no_expression_found"},
        }

    left_raw, operator_token, right_raw = match.group(1), match.group(2), match.group(3)
    left = _parse_number(left_raw)
    right = _parse_number(right_raw)
    operator_func = _OPERATOR_FUNCS[operator_token.lower()]

    try:
        result = operator_func(left, right)
    except ZeroDivisionError:
        return {
            "response": "Non posso dividere per zero.",
            "slots": {},
            "metadata": {"operation": "calculate", "error": "division_by_zero"},
        }

    return {
        "response": f"Fa {_format_result(result)}.",
        "slots": {},
        "metadata": {
            "operation": "calculate",
            "left": left,
            "operator": operator_token,
            "right": right,
            "result": result,
        },
    }
