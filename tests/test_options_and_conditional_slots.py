import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rule_interpreter import RuleInterpreter
from agent.operations.manager import OperationManager


RULES = {
    "create_content": {
        "slots": {
            "content_type": {
                "required": True,
                "type": "string",
                "entity": "CONTENT_TYPE",
                "options": [
                    {"value": "article", "label": "Articolo"},
                    {"value": "tutorial", "label": "Tutorial"},
                ],
            },
            "domain": {
                "required": True,
                "type": "string",
                "entity": "DOMAIN_NAME",
                "options_source": "domains",
            },
            "category": {
                "required": True,
                "type": "string",
                "entity": "CATEGORY_NAME",
                "when": {"content_type": "article"},
            },
            "tutorial": {
                "required": True,
                "type": "string",
                "entity": "TUTORIAL_NAME",
                "when": {"content_type": "tutorial"},
            },
        },
        "default": "__create_content",
        "wait": {
            "content_type": "wait_content_type",
            "domain": "wait_domain",
            "category": "wait_category",
            "tutorial": "wait_tutorial",
        },
        "fallback": "unsupported",
    }
}

RESPONSES = {
    "wait_content_type": ["Vuoi generare articoli o un tutorial?"],
    "wait_domain": ["Per quale dominio?"],
    "wait_category": ["Per quale categoria?"],
    "wait_tutorial": ["Per quale tutorial?"],
    "unsupported": ["Non ho capito."],
}


def _make_interpreter(options_providers=None):
    manager = OperationManager(auto_discover=False)
    for name, provider in (options_providers or {}).items():
        manager._options_providers[name] = provider
    return RuleInterpreter(RULES, RESPONSES, operation_manager=manager)


def test_static_options_attached_to_first_wait():
    interpreter = _make_interpreter()

    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "create_content", {}
    )

    assert wait_slot == "content_type"
    assert bot_slots["__options__"] == [
        {"value": "article", "label": "Articolo"},
        {"value": "tutorial", "label": "Tutorial"},
    ]


def test_dynamic_options_provider_is_called_with_current_slots():
    calls = []

    def domains_provider(slots):
        calls.append(dict(slots))
        return [{"value": "pippo", "label": "Pippo"}]

    interpreter = _make_interpreter({"domains": domains_provider})

    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "create_content", {"content_type": "article"}
    )

    assert wait_slot == "domain"
    assert bot_slots["__options__"] == [{"value": "pippo", "label": "Pippo"}]
    assert calls == [{"content_type": "article"}]


def test_when_clause_skips_irrelevant_slot():
    interpreter = _make_interpreter()

    # content_type=tutorial: "category" non si applica, deve chiedere "tutorial" e non "category"
    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "create_content", {"content_type": "tutorial", "domain": "pippo"}
    )

    assert wait_slot == "tutorial"


def test_when_clause_requires_category_for_article():
    interpreter = _make_interpreter()

    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "create_content", {"content_type": "article", "domain": "pippo"}
    )

    assert wait_slot == "category"


def test_provider_exception_falls_back_to_no_options():
    def broken_provider(slots):
        raise RuntimeError("boom")

    interpreter = _make_interpreter({"domains": broken_provider})

    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "create_content", {"content_type": "article"}
    )

    assert wait_slot == "domain"
    assert "__options__" not in bot_slots
