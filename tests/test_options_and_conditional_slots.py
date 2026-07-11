import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rule_interpreter import RuleInterpreter
from agent.operations.manager import OperationManager


# Fixture generica: testa i meccanismi del DSL (options statiche/dinamiche, slot
# condizionali "when") indipendentemente da un qualsiasi vocabolario di dominio,
# così questo test resta identico su qualunque branch/dominio del progetto.
RULES = {
    "demo_intent": {
        "slots": {
            "kind": {
                "required": True,
                "type": "string",
                "entity": "KIND",
                "options": [
                    {"value": "type_a", "label": "Tipo A"},
                    {"value": "type_b", "label": "Tipo B"},
                ],
            },
            "target": {
                "required": True,
                "type": "string",
                "entity": "TARGET_NAME",
                "options_source": "targets",
            },
            "detail_a": {
                "required": True,
                "type": "string",
                "entity": "DETAIL_A_NAME",
                "when": {"kind": "type_a"},
            },
            "detail_b": {
                "required": True,
                "type": "string",
                "entity": "DETAIL_B_NAME",
                "when": {"kind": "type_b"},
            },
        },
        "default": "__demo_intent",
        "wait": {
            "kind": "wait_kind",
            "target": "wait_target",
            "detail_a": "wait_detail_a",
            "detail_b": "wait_detail_b",
        },
        "fallback": "unsupported",
    }
}

RESPONSES = {
    "wait_kind": ["Che tipo vuoi?"],
    "wait_target": ["Per quale target?"],
    "wait_detail_a": ["Dettaglio A?"],
    "wait_detail_b": ["Dettaglio B?"],
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
        "demo_intent", {}
    )

    assert wait_slot == "kind"
    assert bot_slots["__options__"] == [
        {"value": "type_a", "label": "Tipo A"},
        {"value": "type_b", "label": "Tipo B"},
    ]


def test_dynamic_options_provider_is_called_with_current_slots():
    calls = []

    def targets_provider(slots):
        calls.append(dict(slots))
        return [{"value": "pippo", "label": "Pippo"}]

    interpreter = _make_interpreter({"targets": targets_provider})

    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "demo_intent", {"kind": "type_a"}
    )

    assert wait_slot == "target"
    assert bot_slots["__options__"] == [{"value": "pippo", "label": "Pippo"}]
    assert calls == [{"kind": "type_a"}]


def test_when_clause_skips_irrelevant_slot():
    interpreter = _make_interpreter()

    # kind=type_b: "detail_a" non si applica, deve chiedere "detail_b" e non "detail_a"
    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "demo_intent", {"kind": "type_b", "target": "pippo"}
    )

    assert wait_slot == "detail_b"


def test_when_clause_requires_detail_a_for_type_a():
    interpreter = _make_interpreter()

    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "demo_intent", {"kind": "type_a", "target": "pippo"}
    )

    assert wait_slot == "detail_a"


def test_provider_exception_falls_back_to_no_options():
    def broken_provider(slots):
        raise RuntimeError("boom")

    interpreter = _make_interpreter({"targets": broken_provider})

    response, wait_slot, bot_slots = interpreter.handle_intent_with_bot_slots(
        "demo_intent", {"kind": "type_a"}
    )

    assert wait_slot == "target"
    assert "__options__" not in bot_slots
