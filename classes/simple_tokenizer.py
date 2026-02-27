import re
import os


class SimpleTokenizer:
    def __init__(self, fasttext_model_path=None):
        self.pattern = re.compile(r'[^\w\s]')
        self.fasttext_model = None
        self.vocab = None
        self.word_to_idx = {}
        if fasttext_model_path and os.path.exists(fasttext_model_path):
            import fasttext
            self.fasttext_model = fasttext.load_model(fasttext_model_path)
            self.vocab = self.fasttext_model.words
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
        if self.fasttext_model and word in self.word_to_idx:
            return self.word_to_idx[word]
        return 0
