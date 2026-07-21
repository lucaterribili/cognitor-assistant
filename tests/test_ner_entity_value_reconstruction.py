"""Test della ricostruzione del valore delle entità NER da tokens+tag BIO
(IntentClassifier._extract_entities). Verifica in particolare che caratteri di
punteggiatura interni a un'entità (es. i punti di un nome a dominio) non vengano
sostituiti da spazi: SimpleTokenizer tokenizza "ai.programmato.it" come
["ai", "programmato", "it"] (la punteggiatura diventa spazio prima di tokenizzare),
quindi senza il testo originale il valore ricostruito sarebbe "ai programmato it",
corrompendo il dato prima che arrivi alle Operation (es. il dominio non verrebbe più
trovato nel backoffice)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intellective.intent_classifier import IntentClassifier
from classes.simple_tokenizer import SimpleTokenizer


def _classifier():
    # Bypassa nn.Module.__init__ (richiederebbe wordvectors/vocab reali): i metodi
    # sotto test non usano stato dell'istanza, solo self._token_char_spans (staticmethod).
    return object.__new__(IntentClassifier)


def _bio_tags(tokens, entity_token_indices, entity_type):
    tags = ["O"] * len(tokens)
    for i, idx in enumerate(entity_token_indices):
        tags[idx] = ("B-" if i == 0 else "I-") + entity_type
    return tags


def test_domain_with_dots_preserved_with_original_text():
    clf = _classifier()
    tokenizer = SimpleTokenizer()
    text = "ottimizza i contenuti del dominio ai.programmato.it"
    tokens = tokenizer(text)

    # "ai.programmato.it" tokenizza come 3 token separati (il punto diventa spazio)
    entity_tokens = tokens[-3:]
    assert entity_tokens == ["ai", "programmato", "it"]
    entity_indices = list(range(len(tokens) - 3, len(tokens)))
    ner_tags = _bio_tags(tokens, entity_indices, "DOMAIN_NAME")

    entities = clf._extract_entities(tokens, ner_tags, text)

    assert len(entities) == 1
    assert entities[0]["entity"] == "DOMAIN_NAME"
    assert entities[0]["value"] == "ai.programmato.it"
    # start/end devono essere offset di carattere validi nel testo originale
    start, end = entities[0]["start"], entities[0]["end"]
    assert text.lower()[start:end] == "ai.programmato.it"


def test_domain_with_dots_without_original_text_uses_legacy_token_join():
    clf = _classifier()
    tokenizer = SimpleTokenizer()
    text = "dominio example.com"
    tokens = tokenizer(text)
    entity_indices = list(range(len(tokens) - 2, len(tokens)))
    ner_tags = _bio_tags(tokens, entity_indices, "DOMAIN_NAME")

    entities = clf._extract_entities(tokens, ner_tags)  # nessun original_text

    assert entities[0]["value"] == "example com"  # comportamento legacy invariato
    assert entities[0]["start"] == entity_indices[0]  # indice di token, non carattere
    assert entities[0]["end"] == entity_indices[-1] + 1


def test_multi_word_entity_still_joined_with_space():
    clf = _classifier()
    tokenizer = SimpleTokenizer()
    text = "cerca un articolo su assicurazione auto"
    tokens = tokenizer(text)
    entity_indices = [tokens.index("assicurazione"), tokens.index("auto")]
    ner_tags = _bio_tags(tokens, entity_indices, "POST_QUERY")

    entities = clf._extract_entities(tokens, ner_tags, text)

    assert entities[0]["value"] == "assicurazione auto"


def test_no_entities_returns_empty_list():
    clf = _classifier()
    tokenizer = SimpleTokenizer()
    text = "ciao come stai"
    tokens = tokenizer(text)
    ner_tags = ["O"] * len(tokens)

    assert clf._extract_entities(tokens, ner_tags, text) == []
