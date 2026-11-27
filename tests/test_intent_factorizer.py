import glob
import os

import pytest
import yaml

from classes.intent_factorizer import NO_TYPE, IntentFactorizer
from classes.ner_tag_builder import NERTagBuilder
from config import BASE_DIR

INTENTS_DIR = os.path.join(BASE_DIR, "knowledge", "intents")


# La tassonomia (un file = un argomento) è quella dei dati pre-addestrati, che vivono solo nel branch develop_loc;
# negli altri branch knowledge/intents è il set base e questi test non hanno nulla da verificare.
requires_pretrained_taxonomy = pytest.mark.skipif(
    not os.path.exists(os.path.join(INTENTS_DIR, "affirm.yaml")), reason="tassonomia pre-addestrata assente (solo develop_loc)"
)


def taxonomy():
    """{intento: argomento secondo il file che lo contiene}. In affirm.yaml ogni intento è un argomento a sé."""
    truth = {}
    for path in sorted(glob.glob(os.path.join(INTENTS_DIR, "*.yaml"))):
        stem = os.path.basename(path)[:-5]
        for item in yaml.safe_load(open(path, encoding="utf-8"))["nlu"]["intents"]:
            truth[item["intent"]] = item["intent"] if stem == "affirm" else stem
    return truth


ENTITY_TYPES = ["DEPARTMENT", "FACILITY", "DOCUMENT", "TAX", "HEALTH_SERVICE", "MEDICAL_PROCEDURE", "TECH_CONCEPT"]


def test_splits_type_and_topic():
    names = ["ask_opening_hours", "ask_department_opening_hours", "ask_facility_opening_hours",
             "ask_health_service_opening_hours", "ask_phone", "affirm_greeting"]
    f = IntentFactorizer(names, ENTITY_TYPES)

    assert f.factors_of(0) == ("opening_hours", NO_TYPE)
    assert f.factors_of(1) == ("opening_hours", "DEPARTMENT")
    assert f.factors_of(3) == ("opening_hours", "HEALTH_SERVICE")
    assert f.factors_of(5) == ("affirm_greeting", NO_TYPE)  # gli affirm_* sono un argomento a sé
    assert f.types[0] == NO_TYPE and f.num_topics == 3


def test_a_topic_that_starts_like_a_type_is_not_split():
    # TAX è un tipo di entità, ma tax_return è un argomento: non esiste un argomento "return" da cui staccarlo
    names = ["ask_tax_return", "ask_tax_deductions", "ask_document_tax_return"]
    f = IntentFactorizer(names, ENTITY_TYPES)

    assert f.factors_of(0) == ("tax_return", NO_TYPE)
    assert f.factors_of(1) == ("tax_deductions", NO_TYPE)
    assert f.factors_of(2) == ("tax_return", "DOCUMENT")


@requires_pretrained_taxonomy
def test_every_intent_of_the_real_taxonomy_is_split_like_its_file():
    truth = taxonomy()
    names = list(truth)
    f = IntentFactorizer(names, NERTagBuilder().ENTITY_TYPES)

    wrong = {n: (f.factors_of(i)[0], truth[n]) for i, n in enumerate(names) if f.factors_of(i)[0] != truth[n]}
    assert not wrong, wrong
    assert f.num_topics == len(set(truth.values()))


@requires_pretrained_taxonomy
def test_pairs_identify_every_intent_uniquely():
    names = list(taxonomy())
    f = IntentFactorizer(names, NERTagBuilder().ENTITY_TYPES)

    pairs = list(zip(f.pair_topic, f.pair_type))
    assert len(set(pairs)) == len(names)  # (argomento, tipo) -> un solo intento: la ricostruzione è senza ambiguità


def test_json_roundtrip(tmp_path):
    f = IntentFactorizer(["ask_phone", "ask_department_phone", "affirm_greeting"], ENTITY_TYPES)
    f.save(str(tmp_path / "f.json"))

    g = IntentFactorizer.load(str(tmp_path / "f.json"))

    assert (g.topics, g.types, g.pair_topic, g.pair_type, g.intent_names) == \
           (f.topics, f.types, f.pair_topic, f.pair_type, f.intent_names)


def test_type_applies_only_to_topics_that_have_typed_variants():
    names = ["ask_opening_hours", "ask_department_opening_hours", "ask_phone", "affirm_greeting", "affirm_goodbye"]
    f = IntentFactorizer(names, ENTITY_TYPES)

    assert f.type_applies == [True, True, False, False, False]  # ask_phone non ha varianti con tipo qui; gli affirm_* mai
    g = IntentFactorizer.from_dict(f.to_dict())
    assert g.type_applies == f.type_applies
