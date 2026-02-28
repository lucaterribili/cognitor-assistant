import pytest
from classes.simple_tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer():
    return SimpleTokenizer()


def test_tokenize_simple(tokenizer):
    assert tokenizer.tokenize("Ciao mondo") == ["ciao", "mondo"]


def test_tokenize_empty(tokenizer):
    assert tokenizer.tokenize("") == []


def test_tokenize_spaces(tokenizer):
    assert tokenizer.tokenize("  ciao   mondo  ") == ["ciao", "mondo"]


def test_tokenize_punctuation(tokenizer):
    assert tokenizer.tokenize("ciao, mondo!") == ["ciao", "mondo"]


def test_tokenize_unicode(tokenizer):
    assert tokenizer.tokenize("perché città") == ["perché", "città"]

