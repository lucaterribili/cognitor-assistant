import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from tqdm import tqdm

from config import BASE_DIR, TOKENIZER_PATH
from intellective.intent_classifier import IntentClassifier


class IntentDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_input, output_id = self.data[idx]
        return torch.tensor(tokenized_input, dtype=torch.long), torch.tensor(output_id[0], dtype=torch.long)


def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sentences_padded, labels


def train_model(model, dataloader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_progress = tqdm(dataloader, desc="Training", leave=False)
        model.train()
        total_loss = 0
        for inputs, labels in epoch_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")


def main():
    # Controlla disponibilità di CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    npy_path = os.path.join(BASE_DIR, 'data', 'tokenized_data.npy')
    intent_dict_path = os.path.join(BASE_DIR, 'data', 'intent_dict.json')

    with open(intent_dict_path, 'r') as f:
        intent_dict = json.load(f)
        intents_number = len(intent_dict)

    dataset = IntentDataset(npy_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)

    model = IntentClassifier(
        vocab_size=sp.vocab_size(),
        embed_dim=300,
        hidden_dim=256,
        output_dim=intents_number,
        dropout_prob=0.3,
        sp_model_path=TOKENIZER_PATH,
        fasttext_model_path=os.path.join(BASE_DIR, 'models', 'fasttext_model.bin'),
        freeze_embeddings=True
    )
    model.to(device)

    train_model(model, dataloader, epochs=15, lr=0.001, device=device)

    torch.save(model.state_dict(), "../models/intent_model_fast.pth")

    def predict(sentence_ids):
        model.eval()
        with torch.no_grad():
            tokens = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0).to(device)
            output = model(tokens)
            return torch.argmax(output, dim=1).item()

    # Esempi di predizione
    print(predict([1, 23, 45]))  # Esempio di input tokenizzato
    print(predict([12, 67, 89]))


if __name__ == "__main__":
    main()