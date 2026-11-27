from agent.slot_manager import SlotManager


class _RI:
    def get_valid_values_for_slot(self, intent, slot):
        return ['medicina legale']

    def is_valid_value(self, intent, slot, value):
        return str(value).lower() in ['medicina legale']


def _sm():
    sm = SlotManager.__new__(SlotManager)
    sm.rule_interpreter = _RI()
    sm.get_slot_entity_type = lambda i, s: 'DEPARTMENT'
    return sm


def test_prefers_entity_with_valid_value():
    ents = [{'entity': 'DEPARTMENT', 'value': 'uoc'}, {'entity': 'DEPARTMENT', 'value': 'Medicina Legale'}]
    assert _sm()._extract_from_entities_and_index('i', 'DEPARTMENT', ents) == ('Medicina Legale', 1)


def test_falls_back_to_first():
    ents = [{'entity': 'DEPARTMENT', 'value': 'uoc'}, {'entity': 'DEPARTMENT', 'value': 'altro'}]
    assert _sm()._extract_from_entities_and_index('i', 'DEPARTMENT', ents) == ('uoc', 0)
