import os
import re

import torch
import sentencepiece as spm
from django.conf import settings
from ..classes.splitter import SentenceSplitter
from .neural_network.simple_nn import SentimentModel

class Summarizer:
    _instance = None
    _celery_status = None

    def __new__(cls, *args, **kwargs):
        celery = kwargs.get('celery', False)

        if cls._instance is None or cls._celery_status != celery:
            cls._instance = super(Summarizer, cls).__new__(cls)
            cls._celery_status = celery

        return cls._instance

    def __init__(self, celery=False):
        if not hasattr(self, 'initialized'):
            base_dir = settings.BASE_DIR
            self.model_path = os.path.join(base_dir, 'learning_manage', 'ai_models', 'cutter', 'sentiment_model.pth')
            self.sp_model_path = os.path.join(base_dir, 'learning_manage', 'ai_models', 'tokenizer', 'model.model')
            self.celery = celery
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(self.sp_model_path)

            self.model = SentimentModel(
                vocab_size=self.sp.vocab_size(),
                embed_size=128,
                hidden_size=256,
                output_size=2,
                num_layers=1
            ).to(self.get_device())

            self.model.load_state_dict(torch.load(self.model_path, map_location=self.get_device()))
            self.model.eval()
            self.splitter = SentenceSplitter()
            self.initialized = True

    def analyze(self, text, output_format="string"):
        normalized_text = self.normalize_text(text)
        sentences = self.splitter.get_sentences(text=normalized_text)
        summary = []
        device = self.get_device()

        for sentence in sentences:
            try:
                inputs = self.preprocess_sentence(sentence)
                inputs = torch.tensor(inputs).unsqueeze(0).to(device)

                output = self.model(inputs)
                _, predicted = torch.max(output.data, 1)

                if predicted.item() == 1:
                    summary.append(sentence)
            except ValueError as e:
                print(f"Error processing sentence: {e}")

        if output_format == "list":
            return summary
        else:
            return ' '.join(summary)

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize the input text by removing unwanted characters,
        normalizing spaces, and cleaning up the formatting.

        :param text: The text to normalize.
        :return: The normalized text.
        """

        # Remove any unnecessary newlines and extra spaces
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip()  # Remove leading and trailing whitespace
        text = re.sub(r'\*\s*\*(.*?)\*\s*\*', r'\1', text)  # Convert * * text * * or * *text* * to text
        text = re.sub(r'_(.*?)_', r'\1', text)
        # Remove ?. and !. occurrences
        text = re.sub(r'\?\.', '?', text)  # Replace '?.' with '?'
        text = re.sub(r'!\.', '!', text)  # Replace '!.' with '!'
        # Remove final period if present
        text = re.sub(r'\.$', '', text)
        # Normalize quotation marks and apostrophes
        text = text.replace('«', '"').replace('»', '"').replace('“', '"')
        text = text.replace("’", "'")  # Normalize apostrophes

        return text

    def preprocess_sentence(self, sentence):
        indices = self.sp.Encode(sentence, out_type=int)
        if not indices:
            raise ValueError(f"Empty input for sentence: '{sentence}'")
        return indices

    def get_device(self):
        current_device = 'cpu' if self.celery else ('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(current_device)