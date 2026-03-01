import random
from typing import Any


class AnswerManager:
    def __init__(self, rules: dict):
        self.rules = rules

    def set_slot(self, slots: dict, key: str, value: Any):
        slots[key] = value

    def _check_condition(self, condition: dict, slots: dict) -> bool:
        slot_value = slots.get(condition["slot"])
        operator = condition["operator"]
        expected_value = condition.get("value")

        if operator == "filled":
            return slot_value is not None
        if operator == "not_filled":
            return slot_value is None
        if operator in ("eq", "neq"):
            if isinstance(slot_value, str) and isinstance(expected_value, str):
                slot_lower = slot_value.lower()
                expected_lower = expected_value.lower()
                return slot_lower == expected_lower if operator == "eq" else slot_lower != expected_lower
            return slot_value == expected_value if operator == "eq" else slot_value != expected_value
        if operator == "gt":
            return slot_value is not None and slot_value > expected_value
        if operator == "lt":
            return slot_value is not None and slot_value < expected_value
        if operator == "contains":
            if isinstance(slot_value, str) and isinstance(expected_value, str):
                return expected_value.lower() in slot_value.lower()
            return expected_value in (slot_value or "")
        return False

    def resolve(self, intent: str, slots: dict) -> dict:
        rule = self.rules.get(intent)
        if not rule:
            return {"response": "default_fallback", "wait_for_slot": None}

        for branch in rule.get("conditions", []):
            if all(self._check_condition(condition, slots) for condition in branch["if"]):
                return {
                    "response": branch["response"],
                    "wait_for_slot": branch.get("wait_for_slot")
                }

        return {"response": rule.get("default", "default_fallback"), "wait_for_slot": None}

    def get_response(self, intent: str, slots: dict, responses: dict) -> tuple[str, str | None]:
        resolved = self.resolve(intent, slots)
        response_key = resolved["response"]
        wait_for_slot = resolved["wait_for_slot"]

        response_list = responses.get(response_key, [])
        if not response_list:
            return f"Risposta non definita per {response_key}", None

        return random.choice(response_list), wait_for_slot
