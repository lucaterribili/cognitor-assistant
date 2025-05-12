import json
import os
import torch
from django.core.management.base import BaseCommand
from django.conf import settings
from ...ai_models.neural_network.simple_nn import SentimentModel
from ...classes.splitter import SentenceSplitter
import sentencepiece as spm  # Import SentencePiece

from ...helpers.utility import get_model


class Command(BaseCommand):
    help = 'Use trained model to predict sentence importance'

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def handle(self, *args, **kwargs):
        document_model = get_model('Document')
        base_dir = settings.BASE_DIR
        model_path = os.path.join(base_dir, 'learning_manage', 'ai_models', 'cutter', 'sentiment_model.pth')
        sp_model_path = os.path.join(base_dir, 'learning_manage', 'ai_models', 'tokenizer',
                                     'model.model')

        # Carica il modello SentencePiece
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        model = SentimentModel(vocab_size=sp.get_piece_size(), embed_size=128, hidden_size=256, output_size=2,
                               num_layers=1).to(self.device)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        documents = document_model.objects.filter(pk=444)
        splitter = SentenceSplitter()
        for document in documents:
            sentences = splitter.get_sentences(text=document.content)
            summary = []
            for sentence in sentences:
                try:
                    inputs = self.preprocess_sentence(sentence, sp)

                    inputs = torch.tensor(inputs).unsqueeze(0).to(self.device)

                    output = model(inputs)
                    _, predicted = torch.max(output.data, 1)

                    if predicted.item() == 1:
                        print(f'{sentence} : IMPORTANTE')
                    else:
                        print(f'{sentence} : CAZZATA')
                except ValueError as e:
                    print(e)

    @staticmethod
    def preprocess_sentence(sentence, sp):
        indices = sp.encode(sentence, out_type=int)

        if not indices:
            raise ValueError(f"Empty input for sentence: '{sentence}'")  # Raise error for empty input

        return indices  # Restituisci gli indici senza convertirli in tensor qui
