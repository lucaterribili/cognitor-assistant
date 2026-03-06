import re
import os
import json


class SimpleTokenizer:
    def __init__(self, vocab_path=None):
        """
        Tokenizer semplice che usa solo il vocabolario.

        Args:
            vocab_path: path al file vocab.json (default: .cognitor/vocab.json)
        """
        self.pattern = re.compile(r'[^\w\s]')
        self.vocab = None
        self.word_to_idx = {}

        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

    def tokenize(self, text):
        text = text.lower()
        text = self.pattern.sub(' ', text)
        tokens = text.split()
        return tokens

    def __call__(self, text):
        return self.tokenize(text)

    def __len__(self):
        return len(self.vocab) if self.vocab else 0

    def get_word_index(self, word):
        """Ritorna l'indice della parola nel vocabolario, 0 se non trovata."""
        return self.word_to_idx.get(word, 0)
