import os
import pandas as pd
import sentencepiece as spm
from django.core.management.base import BaseCommand
from django.conf import settings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from ...ai_models.neural_network.simple_nn import SentimentDataset, SentimentModel


class Command(BaseCommand):
    help = 'Train Model Cut Fluff'

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def handle(self, *args, **kwargs):
        base_dir = settings.BASE_DIR
        csv_file = os.path.join(base_dir.parent, 'dataset', 'sentences.csv')
        df = pd.read_csv(csv_file)
        sentences = df['SENTENCE'].values
        labels = df['IMPORTANCE'].values

        # Addestramento di SentencePiece
        self.train_sentencepiece(sentences)
        # Carica il modello SentencePiece
        sp_model_path = os.path.join(base_dir, 'learning_manage', 'ai_models', 'sentencepiece', 'model.model')
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        # Tokenizzazione delle frasi
        tokenized_sentences = [sp.encode(sentence, out_type=int) for sentence in sentences]

        # Padding e creazione dei dataset
        train_sentences, val_sentences, train_labels, val_labels = train_test_split(
            tokenized_sentences, labels, test_size=0.2, random_state=42)

        train_dataset = SentimentDataset(train_sentences, train_labels)
        val_dataset = SentimentDataset(val_sentences, val_labels)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Impostazione dei parametri del modello
        vocab_size = sp.get_piece_size()  # Ottieni la dimensione del vocabolario da SentencePiece
        embed_size = 128
        hidden_size = 256
        output_size = 2  # Classificazione binaria (0 o 1)
        num_layers = 1

        model = SentimentModel(vocab_size, embed_size, hidden_size, output_size, num_layers).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Addestramento del modello
        self.train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=30)

        # Salva il modello
        self.save_model(model, 'sentiment_model.pth')

    def train_sentencepiece(self, sentences):
        # Definisci i percorsi per il file di input e il modello
        base_dir = settings.BASE_DIR  # Ottieni la directory base del progetto
        sentences_file_path = os.path.join(base_dir, 'learning_manage', 'ai_models', 'tokenizer', 'sentences.txt')
        model_prefix = os.path.join(base_dir, 'learning_manage', 'ai_models', 'tokenizer', 'model')

        # Crea la cartella se non esiste
        os.makedirs(os.path.dirname(sentences_file_path), exist_ok=True)

        # Salva le frasi in un file di testo
        with open(sentences_file_path, 'w') as f:
            for sentence in sentences:
                f.write(sentence + '\n')

        # Configurazione dell'addestramento di SentencePiece
        spm.SentencePieceTrainer.Train(
            f'--input={sentences_file_path} --model_prefix={model_prefix} --vocab_size=5000 --character_coverage=1.0')

        print('SentencePiece model trained and saved.')

    def train_model(self, model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}')

            # Valutazione sul set di validazione
            self.evaluate_model(model, val_loader)

    def evaluate_model(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f'Validation Accuracy: {val_acc:.4f}')

    def save_model(self, model, model_path):
        """Save the trained model to a file."""
        base_dir = settings.BASE_DIR
        target_path = os.path.join(base_dir, 'learning_manage', 'ai_models', 'cutter', model_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(model.state_dict(), str(target_path))
        print(f'Model saved to {model_path}')