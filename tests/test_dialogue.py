import json

import pytest

from dialogue.actions import ACTIONS, Action
from dialogue.dev_set import evaluate_dev_set, load_dev_set
from dialogue.rules import RAG_HIGH, RAG_LOW, RAG_NONE, decide, decide_from_observation
from dialogue.simulator import Simulator, baseline_predictions, write_jsonl
from dialogue.state import DialogueState
from dialogue.taxonomy import Taxonomy


@pytest.fixture(scope="module")
def tax():
    return Taxonomy(
        allowed_types={
            "opening_hours": ["DEPARTMENT", "FACILITY"], "phone": ["DEPARTMENT", "FACILITY"], "help": ["MISC"],
            "contact_human": [], "clarification": [], "affirm_greeting": [], "affirm_goodbye": [],
            "affirm_gratitude": [], "affirm_insult": [], "affirm_confirmation": [], "affirm_negation": [],
        },
        answerable_without_entity={"help"}, handoff_topics={"contact_human"}, repeat_topics={"clarification"},
        affirm_topics={"greeting": "affirm_greeting", "goodbye": "affirm_goodbye", "thanks": "affirm_gratitude",
                       "insult": "affirm_insult", "confirm": "affirm_confirmation", "deny": "affirm_negation"},
    )


def state(last=None, context=None, memory=(), unclear=0):
    return DialogueState(last_action=Action(last) if last else None, context_topic=context,
                         memory=set(memory), unclear_count=unclear)


# ---------- regole (oracolo) ----------

@pytest.mark.parametrize("topic, entities, rag, st, expected", [
    ("opening_hours", [], RAG_HIGH, state(), Action.ASK_SLOT),                      # serve l'entità e non c'è
    ("opening_hours", ["DEPARTMENT"], RAG_HIGH, state(), Action.ANSWER),
    ("opening_hours", [], RAG_HIGH, state(memory=["FACILITY"]), Action.ANSWER),     # la memoria basta
    ("opening_hours", ["DATE"], RAG_HIGH, state(), Action.ASK_SLOT),                # entità di tipo non pertinente
    ("help", [], RAG_HIGH, state(), Action.ANSWER),                                 # argomento generico: nessuna entità
    ("opening_hours", ["DEPARTMENT"], RAG_NONE, state(), Action.FALLBACK),
    ("opening_hours", ["DEPARTMENT"], RAG_LOW, state(), Action.CLARIFY),
    ("opening_hours", ["DEPARTMENT"], RAG_LOW, state(unclear=1), Action.FALLBACK),
    (None, ["FACILITY"], RAG_HIGH, state(last="ask_slot", context="phone"), Action.ANSWER),   # risposta allo slot
    (None, ["FACILITY"], RAG_HIGH, state(last="answer", context="opening_hours"), Action.ANSWER),  # follow-up
    (None, [], RAG_NONE, state(), Action.CLARIFY),
    (None, [], RAG_NONE, state(unclear=1), Action.FALLBACK),
    ("affirm_greeting", [], RAG_HIGH, state(), Action.GREET),
    ("affirm_goodbye", [], RAG_HIGH, state(), Action.GOODBYE),
    ("affirm_gratitude", [], RAG_HIGH, state(), Action.THANKS_REPLY),
    ("affirm_insult", [], RAG_HIGH, state(), Action.DEESCALATE),
    ("contact_human", [], RAG_HIGH, state(), Action.HANDOFF),
    ("affirm_confirmation", [], RAG_HIGH, state(last="offer_more"), Action.PROMPT),
    ("affirm_confirmation", [], RAG_HIGH, state(last="answer"), Action.OFFER_MORE),
    ("affirm_confirmation", [], RAG_HIGH, state(last="clarify", context="phone"), Action.ANSWER),
    ("affirm_negation", [], RAG_HIGH, state(last="offer_more"), Action.GOODBYE),
    ("affirm_negation", [], RAG_HIGH, state(last="answer"), Action.CLARIFY),
    ("clarification", [], RAG_HIGH, state(last="answer"), Action.REPEAT),
    ("clarification", [], RAG_HIGH, state(), Action.CLARIFY),
])
def test_rules(tax, topic, entities, rag, st, expected):
    assert decide(topic, entities, rag, st, tax) == expected


def test_baseline_uses_the_confidence_threshold(tax):
    obs = {"topic": "opening_hours", "topic_conf": 0.3, "entity_types": ["DEPARTMENT"], "rag": RAG_HIGH}

    assert decide_from_observation(obs, state(), tax) == Action.CLARIFY  # argomento sotto soglia = sconosciuto
    assert decide_from_observation({**obs, "topic_conf": 0.9}, state(), tax) == Action.ANSWER


# ---------- stato ----------

def test_state_tracks_pending_slot_memory_and_unclear_turns():
    st = DialogueState()

    st.update("phone", [], Action.ASK_SLOT)
    assert st.pending and st.context_topic == "phone"

    st.update(None, ["FACILITY"], Action.ANSWER)
    assert not st.pending and st.context_topic == "phone" and st.memory == {"FACILITY"}  # l'argomento resta in contesto

    st.update(None, [], Action.CLARIFY)
    st.update(None, [], Action.FALLBACK)
    assert st.unclear_count == 2
    st.update("affirm_greeting", [], Action.GREET)
    assert st.unclear_count == 0 and st.last_action == Action.GREET
    assert len(st.history) <= 5


def test_clarify_keeps_the_topic_so_a_yes_can_be_resolved(tax):
    st = DialogueState()
    st.update("opening_hours", ["DEPARTMENT"], Action.CLARIFY)

    assert st.context_topic == "opening_hours"
    assert decide("affirm_confirmation", [], RAG_HIGH, st, tax) == Action.ANSWER


# ---------- generatore ----------

@pytest.fixture(scope="module")
def real_taxonomy():
    try:
        return Taxonomy.load()
    except FileNotFoundError:
        pytest.skip("manca .cognitor/intent_factors.json (esegui il training)")


def test_generation_is_deterministic_and_seed_dependent(real_taxonomy):
    a = list(Simulator(real_taxonomy, seed=3).generate(30))
    b = list(Simulator(real_taxonomy, seed=3).generate(30))
    c = list(Simulator(real_taxonomy, seed=4).generate(30))

    assert a == b and a != c


def test_generated_dialogues_are_well_formed_and_cover_all_actions(real_taxonomy):
    turns = list(Simulator(real_taxonomy, seed=1).generate(600))

    assert {t["action"] for t in turns} == set(ACTIONS)
    per_dialogue = {}
    for t in turns:
        per_dialogue.setdefault(t["dialogue"], []).append(t["turn"])
    assert all(v == list(range(len(v))) and len(v) <= 10 for v in per_dialogue.values())
    assert all(t["state"]["last_action"] is None for t in turns if t["turn"] == 0)  # si parte da zero


def test_ellipsis_and_slot_reply_turns_exist(real_taxonomy):
    """Le conversazioni devono contenere i casi in cui serve il contesto: l'argomento non è nel messaggio."""
    turns = list(Simulator(real_taxonomy, noise=0.0, seed=2).generate(600))

    slot_replies = [t for t in turns if t["move"] == "slot_reply"]
    follow_ups = [t for t in turns if t["move"] == "follow_up_entity"]
    assert slot_replies and follow_ups
    assert all(t["action"] in {"answer", "fallback", "clarify"} for t in slot_replies + follow_ups)
    assert all(t["state"]["context_topic"] for t in slot_replies + follow_ups)


def test_noise_makes_the_baseline_worse(real_taxonomy):
    def accuracy(noise):
        turns = list(Simulator(real_taxonomy, noise=noise, seed=7).generate(500))
        predicted = baseline_predictions(turns, real_taxonomy)
        return sum(p == t["action"] for p, t in zip(predicted, turns)) / len(turns)

    clean, mild, heavy = accuracy(0.0), accuracy(0.15), accuracy(0.3)

    assert clean > 0.93  # senza rumore resta solo l'ambiguità vera (fuori argomento / incomprensibile)
    assert clean > mild > heavy


def test_jsonl_roundtrip(real_taxonomy, tmp_path):
    turns = list(Simulator(real_taxonomy, seed=5).generate(20))
    path = tmp_path / "d.jsonl"

    assert write_jsonl(iter(turns), str(path)) == len(turns)
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == turns


# ---------- set di sviluppo scritto a mano ----------

def test_dev_set_is_valid_and_the_baseline_handles_most_of_it(real_taxonomy):
    dev = load_dev_set()
    assert len(dev) >= 8
    topics = set(real_taxonomy.topics) | {None}
    for dialogue in dev:
        for turn in dialogue["turns"]:
            assert turn["topic"] in topics, turn
            assert turn["action"] in ACTIONS, turn

    accuracy, misses = evaluate_dev_set(
        lambda obs, st: decide_from_observation(obs, st, real_taxonomy).value, real_taxonomy)

    assert accuracy >= 0.9, misses
