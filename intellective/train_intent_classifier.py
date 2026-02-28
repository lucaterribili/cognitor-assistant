import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import fasttext
from tqdm import tqdm

from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier


class IntentDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_input = item[0]
        output_id = item[1][0]
        ner_tags = item[2] if len(item) > 2 else [0] * len(tokenized_input)  # Default: tutti tag O

        return (
            torch.tensor(tokenized_input, dtype=torch.long),
            torch.tensor(output_id, dtype=torch.long),
            torch.tensor(ner_tags, dtype=torch.long)
        )


def collate_fn(batch):
    sentences, labels, ner_tags_list = zip(*batch)

    # Padding delle sequenze
    sentences_padded = pad_sequence(list(sentences), batch_first=True, padding_value=0)
    ner_tags_padded = pad_sequence(list(ner_tags_list), batch_first=True, padding_value=0)

    # Crea maschere di padding (True per token validi, False per padding)
    masks = pad_sequence([torch.ones(len(s), dtype=torch.bool) for s in sentences],
                        batch_first=True, padding_value=False)

    labels = torch.stack(labels)

    return sentences_padded, labels, ner_tags_padded, masks


def train_model(model, dataloader, epochs, lr, device, intent_weight=1.0, ner_weight=0.5, patience=10):
    intent_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_progress = tqdm(dataloader, desc="Training", leave=False)
        model.train()
        total_loss = 0
        total_intent_loss = 0
        total_ner_loss = 0

        for inputs, intent_labels, ner_tags, masks in epoch_progress:
            inputs = inputs.to(device)
            intent_labels = intent_labels.to(device)
            ner_tags = ner_tags.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass con NER tags per training
            intent_logits, ner_loss = model(inputs, ner_tags=ner_tags, mask=masks)

            # Calcola loss intent
            intent_loss = intent_criterion(intent_logits, intent_labels)

            # Loss combinato pesato
            loss = intent_weight * intent_loss + ner_weight * ner_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_intent_loss += intent_loss.item()
            total_ner_loss += ner_loss.item()

            epoch_progress.set_postfix(
                loss=loss.item(),
                intent_loss=intent_loss.item(),
                ner_loss=ner_loss.item()
            )

        avg_loss = total_loss / len(dataloader)
        avg_intent_loss = total_intent_loss / len(dataloader)
        avg_ner_loss = total_ner_loss / len(dataloader)

        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Intent Loss: {avg_intent_loss:.4f}, NER Loss: {avg_ner_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ Miglioramento! Nuovo best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento. Pazienza: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping attivato dopo {epoch + 1} epoche")
                print(f"Best loss: {best_loss:.4f}")
                # Ripristina il miglior modello
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("✓ Ripristinato il miglior modello")
                break


def train_main_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    npy_path = os.path.join(BASE_DIR, 'data', 'tokenized_data.npy')
    intent_dict_path = os.path.join(BASE_DIR, 'data', 'intent_dict.json')

    with open(intent_dict_path, 'r') as f:
        intent_dict = json.load(f)
        intents_number = len(intent_dict)

    dataset = IntentDataset(npy_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
    ft_model = fasttext.load_model(fasttext_model_path)
    vocab_size = len(ft_model.words)

    model = IntentClassifier(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=256,
        output_dim=intents_number,
        dropout_prob=0.3,
        fasttext_model_path=fasttext_model_path,
        freeze_embeddings=True
    )
    model.to(device)

    train_model(model, dataloader, epochs=50, lr=0.001, device=device)

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'models', 'intent_model_fast.pth'))

    def predict(sentence_ids):
        model.eval()
        with torch.no_grad():
            tokens = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0).to(device)
            output = model(tokens)
            return torch.argmax(output, dim=1).item()
