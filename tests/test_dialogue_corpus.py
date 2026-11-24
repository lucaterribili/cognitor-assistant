import random
import textwrap

import pytest

from dialogue.actions import ACTIONS, Move
from dialogue.corpus import load_corpus, stats, validate
from dialogue.flow_model import FlowModel, SiteVocabulary, instantiate, user_token
from dialogue.taxonomy import Taxonomy


@pytest.fixture(scope="module")
def taxonomy():
    try:
        return Taxonomy.load(config_path=None)
    except FileNotFoundError:
        pytest.skip("manca .cognitor/intent_factors.json (esegui il training)")


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def write(tmp_path, body):
    path = tmp_path / "c.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_corpus(str(path))


# ---------- il corpus vero ----------

def test_the_seed_corpus_is_valid(corpus, taxonomy):
    assert validate(corpus, taxonomy) == []


def test_the_seed_corpus_covers_every_action_and_move(corpus):
    s = stats(corpus)

    assert s["conversations"] >= 50 and s["turns"] >= 190
    assert s["actions_never_seen"] == [] and s["moves_never_seen"] == []
    assert len(s["topics"]) >= 20 and len(s["entity_types"]) >= 6


def test_conversation_ids_are_unique(corpus):
    ids = [c.id for c in corpus]
    assert len(ids) == len(set(ids))


# ---------- il controllo individua gli errori ----------

BAD = """
    conversations:
      - id: bad
        source: llm
        turns:
          - {u: "x", move: question, topic: argomento_inventato, bot: answer}
          - {u: "x", move: question, topic: phone, ent: [TIPO_INVENTATO], bot: azione_inventata}
          - {u: "x", move: slot_reply, ent: [FACILITY], bot: answer}
          - {u: "x", move: question, topic: phone, carry: [DEPARTMENT], bot: answer}
          - {u: "x", move: greeting, topic: phone, bot: greet}
          - {u: "x", move: question, topic: phone, rag: boh, bot: answer}
"""


def test_validation_reports_each_kind_of_error(tmp_path, taxonomy):
    errors = "\n".join(validate(write(tmp_path, BAD), taxonomy))

    for expected in ("argomento 'argomento_inventato'", "tipo di entità 'TIPO_INVENTATO'", "azione 'azione_inventata'",
                     "slot_reply senza", "carry 'DEPARTMENT'", "la mossa 'greeting' non ha un argomento",
                     "rag 'boh'"):
        assert expected in errors, expected


def test_validation_accepts_a_slot_reply_after_an_interruption(tmp_path, taxonomy):
    corpus = write(tmp_path, """
        conversations:
          - id: ok
            turns:
              - {u: "Come prenoto?", move: question, topic: booking, bot: ask_slot}
              - {u: "Siete inutili", move: insult, bot: deescalate}
              - {u: "Una visita", move: slot_reply, ent: [MEDICAL_PROCEDURE], bot: answer}
    """)

    assert validate(corpus, taxonomy) == []


def test_duplicate_ids_are_reported(tmp_path, taxonomy):
    corpus = write(tmp_path, """
        conversations:
          - {id: a, turns: [{u: "Ciao", move: greeting, bot: greet}]}
          - {id: a, turns: [{u: "Ciao", move: greeting, bot: greet}]}
    """)

    assert any("duplicato" in e for e in validate(corpus, taxonomy))


# ---------- modello dei flussi ----------

def test_flow_model_samples_are_deterministic_and_valid(corpus):
    model = FlowModel.fit(corpus)

    a = [model.sample(random.Random(5)) for _ in range(3)]
    b = [model.sample(random.Random(5)) for _ in range(3)]

    assert a == b
    for flow in model.sample(random.Random(1)), model.sample(random.Random(2)):
        assert 1 <= len(flow) <= 12
        assert all(bot in ACTIONS for _, bot in flow)


def test_sampled_flows_are_coherent_slot_replies_follow_a_question_for_a_slot(corpus):
    model = FlowModel.fit(corpus)
    rng = random.Random(0)

    for _ in range(400):
        pending = False
        for token, bot in model.sample(rng):
            if token.startswith(Move.SLOT_REPLY.value):
                assert pending, "risposta a uno slot che nessuno ha chiesto"
            if bot == "ask_slot":
                pending = True
            elif bot == "answer":
                pending = False


def test_flow_model_reproduces_what_the_corpus_shows(corpus):
    """Nel corpus a un saluto segue sempre `greet` e a un ringraziamento `thanks_reply`: il modello lo impara."""
    model = FlowModel.fit(corpus)

    assert set(model.bot_given_user["greeting"]) == {"greet"}
    assert set(model.bot_given_user["thanks"]) == {"thanks_reply"}
    assert "ask_slot" in model.bot_given_user["question"]  # domanda senza entità: serve chiedere


def test_changing_the_corpus_changes_the_model(tmp_path, taxonomy):
    """Il comportamento sta nei dati: un corpus in cui il bot risponde sempre al saluto con `fallback` lo cambia."""
    corpus = write(tmp_path, """
        conversations:
          - {id: x1, turns: [{u: "Ciao", move: greeting, bot: fallback}]}
          - {id: x2, turns: [{u: "Salve", move: greeting, bot: fallback}]}
    """)

    assert set(FlowModel.fit(corpus).bot_given_user["greeting"]) == {"fallback"}


def test_json_roundtrip(corpus):
    model = FlowModel.fit(corpus)

    again = FlowModel.from_json(model.to_json())

    assert again.sample(random.Random(9)) == model.sample(random.Random(9))


def test_user_token_marks_entities_carry_and_retrieval(corpus):
    from dialogue.corpus import Turn

    assert user_token(Turn("x", "question", "answer", topic="phone", entities=["FACILITY"])) == "question+ent"
    assert user_token(Turn("x", "question", "answer", topic="phone", carry=["FACILITY"])) == "question+carry"
    assert user_token(Turn("x", "question", "clarify", topic="phone", retrieval="low")) == "question+rag_low"


# ---------- istanziazione con il vocabolario del sito ----------

def test_instantiation_uses_the_site_vocabulary_and_stays_consistent(corpus, taxonomy):
    model = FlowModel.fit(corpus)
    vocab = SiteVocabulary(
        pairs=[("opening_hours", "DEPARTMENT"), ("opening_hours", "FACILITY"), ("phone", "DEPARTMENT"),
               ("phone", ""), ("help", "")],
        values={"DEPARTMENT": ["radiologia", "cardiologia"], "FACILITY": ["CUP"]},
    )
    rng = random.Random(3)
    seen_entities = False

    for _ in range(300):
        turns = instantiate(model, model.sample(rng), vocab, rng)
        memory = []
        for t in turns:
            if t["topic"]:
                assert t["topic"] in vocab.topics
            for e in t["entities"]:
                seen_entities = True
                assert e["value"] in vocab.values[e["type"]]
                assert e["type"] in vocab.types_of(t["topic"])   # il tipo è ammesso dall'argomento
                memory.append(e["type"])
            for c in t["carry"]:
                assert c in memory                                # il contesto riprende un'entità già nominata
            if t["move"] == "slot_reply":
                assert t["entities"] and t["topic"]
    assert seen_entities


def test_instantiation_without_site_values_uses_placeholders(taxonomy, corpus):
    model = FlowModel.fit(corpus)
    vocab = SiteVocabulary.from_taxonomy(taxonomy)

    turns = [t for _ in range(50) for t in instantiate(model, model.sample(random.Random(_)), vocab, random.Random(_))]

    assert any(e["value"].startswith("<") for t in turns for e in t["entities"])
