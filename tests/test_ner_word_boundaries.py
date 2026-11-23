import os

import pytest

from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier


def extract(tokens, tags, text):
    # _extract_entities non usa lo stato del modello: non serve caricarlo
    return IntentClassifier.__new__(IntentClassifier)._extract_entities(tokens, tags, text)


def test_entity_cut_in_the_middle_of_a_word_is_extended():
    text = "come si rinnova la carta d'identità presso il comune?"
    tokens = ["come", "si", "rinno", "va", "la", "carta", "d", "'", "i", "dentità", "pre", "sso", "il", "comune", "?"]
    tags = ["O"] * 5 + ["B-DOCUMENT", "I-DOCUMENT", "I-DOCUMENT", "I-DOCUMENT"] + ["O"] * 6

    [entity] = extract(tokens, tags, text)

    assert entity["value"] == "carta d'identità"
    assert (entity["start"], entity["end"]) == (19, 35)


def test_entity_starting_in_the_middle_of_a_word_is_extended_backwards():
    text = "pagamento del ticket"
    tokens = ["paga", "mento", "del", "tic", "ket"]
    tags = ["O", "O", "O", "O", "B-DOCUMENT"]  # il primo pezzo della parola è O

    [entity] = extract(tokens, tags, text)

    assert entity["value"] == "ticket"


def test_entity_already_on_word_boundaries_is_unchanged():
    text = "dove trovo il modulo dell'esenzione?"
    tokens = ["dove", "trovo", "il", "modulo", "dell", "'", "esenzione", "?"]
    tags = ["O"] * 6 + ["B-DOCUMENT", "O"]

    [entity] = extract(tokens, tags, text)

    assert entity["value"] == "esenzione"


def test_offsets_beyond_the_text_do_not_crash():
    # il tokenizer ha normalizzato il testo ("…" -> "..."): il token non si ritrova e l'offset sfora la stringa
    text = "orari del pronto soccorso…"
    tokens = ["orari", "del", "pronto", "soccorso", "...", "extra"]
    tags = ["O", "O", "O", "O", "O", "B-DEPARTMENT"]

    entities = extract(tokens, tags, text)

    assert len(entities) == 1
    assert 0 <= entities[0]["start"] <= entities[0]["end"] <= len(text)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(BASE_DIR, "models", "intent_model_fast.pth")),
    reason="modello addestrato non presente",
)
def test_entities_of_trained_model_are_on_word_boundaries():
    import csv

    import torch

    from agent.model_loader import ModelLoader

    loader = ModelLoader(BASE_DIR, torch.device("cpu"))
    intent_dict = loader.load_intent_dict()
    model, loaded = loader.load_intent_classifier(loader.load_vocab_size(), len(intent_dict))
    assert loaded

    with open(os.path.join(BASE_DIR, ".cognitor", "nlu_data.csv"), encoding="utf-8") as f:
        sentences = [row[0] for row in list(csv.reader(f))[1:600]]
    sentences.append("come si rinnova la carta d'identità presso il comune?")

    for sentence in sentences:
        result = model.predict(sentence)
        for e in model._extract_entities(result["tokens"], result["ner_tags"], sentence):
            text = sentence.lower()
            assert e["start"] == 0 or not (text[e["start"] - 1].isalnum() and text[e["start"]].isalnum()), (sentence, e)
            assert e["end"] == len(text) or not (text[e["end"]].isalnum() and text[e["end"] - 1].isalnum()), (sentence, e)
