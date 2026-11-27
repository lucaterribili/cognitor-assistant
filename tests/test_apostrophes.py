"""Gli apostrofi si eliminano sempre (training e inferenza): vedi classes/text_normalizer.py."""
from agent.model_loader import KnowledgeLoader
from agent.rule_interpreter import RuleInterpreter
from classes.ner_markup_parser import NERMarkupParser
from classes.text_normalizer import comparable, strip_apostrophes


def test_strip_apostrophes_keeps_length_and_all_variants():
    text = "dell'UOC dell’ospedale po` un´ʼ"
    out = strip_apostrophes(text)
    assert len(out) == len(text)
    assert not any(c in out for c in "'’`´ʼ‘")
    assert out.startswith("dell UOC dell ospedale")


def test_comparable_ignores_case_apostrophes_and_spaces():
    assert comparable("Sant'Anna  Hospital") == comparable("sant anna hospital")


def test_parser_offsets_match_text_without_apostrophes():
    clean, entities = NERMarkupParser().parse("Quali sono gli orari dell'[UOC Cardiologia](DEPARTMENT) e UTIC?")
    assert "'" not in clean
    [e] = entities
    assert clean[e["start"]:e["end"]] == e["value"] == "UOC Cardiologia"


def test_rule_case_matches_value_extracted_from_text_without_apostrophes():
    rules = {"ask_hours": {"slots": {"DEPARTMENT": {"required": True}}, "wait": "w", "fallback": "fb",
                           "cases": {"ospedale sant'anna": "r_anna"}}}
    responses = {"w": ["quale?"], "fb": ["non so"], "r_anna": ["8-13"]}
    interpreter = RuleInterpreter(rules, responses)
    assert interpreter.is_valid_value("ask_hours", "DEPARTMENT", "ospedale sant anna")
    response = interpreter.handle_intent_with_bot_slots("ask_hours", {"DEPARTMENT": "ospedale sant anna"})[0]
    assert response == "8-13"


def test_missing_intents_dir_does_not_stop_loading_when_doping_off(tmp_path):
    loader = KnowledgeLoader(str(tmp_path))
    loader.build_doping_lookup_table(object())  # senza knowledge/intents: nessuna eccezione
