import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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


def train_with_validation(model, train_dataloader, val_dataloader, epochs, lr, device, intent_weight=1.0, ner_weight=0.5, patience=10):
    intent_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_dataloader, desc="Training", leave=False)
        for inputs, intent_labels, ner_tags, masks in train_progress:
            inputs, intent_labels, ner_tags, masks = inputs.to(device), intent_labels.to(device), ner_tags.to(device), masks.to(device)
            optimizer.zero_grad()
            intent_logits, ner_loss = model(inputs, ner_tags=ner_tags, mask=masks)
            intent_loss = intent_criterion(intent_logits, intent_labels)
            loss = intent_weight * intent_loss + ner_weight * ner_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_intent_loss = 0
        total_val_ner_loss = 0
        with torch.no_grad():
            for inputs, intent_labels, ner_tags, masks in val_dataloader:
                inputs, intent_labels, ner_tags, masks = inputs.to(device), intent_labels.to(device), ner_tags.to(device), masks.to(device)
                intent_logits, ner_loss = model(inputs, ner_tags=ner_tags, mask=masks)
                intent_loss = intent_criterion(intent_logits, intent_labels)
                loss = intent_weight * intent_loss + ner_weight * ner_loss
                total_val_loss += loss.item()
                total_val_intent_loss += intent_loss.item()
                total_val_ner_loss += ner_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_intent_loss = total_val_intent_loss / len(val_dataloader)
        avg_val_ner_loss = total_val_ner_loss / len(val_dataloader)

        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Intent: {avg_val_intent_loss:.4f}, NER: {avg_val_ner_loss:.4f})")

        # Early stopping check on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ Miglioramento Val Loss! Nuovo best: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Pazienza: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping dopo {epoch + 1} epoche")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("✓ Ripristinato il miglior modello basato sulla validation loss")


def train_main_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    npy_path = os.path.join(BASE_DIR, '.cognitor', 'tokenized_data.npy')
    intent_dict_path = os.path.join(BASE_DIR, '.cognitor', 'intent_dict.json')

    with open(intent_dict_path, 'r') as f:
        intent_dict = json.load(f)
        intents_number = len(intent_dict)

    full_dataset = IntentDataset(npy_path)
    
    # Split train/validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Carica vocab_size da vocab.json invece di FastText
    vocab_path = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    wordvectors_path = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')

    model = IntentClassifier(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=256,
        output_dim=intents_number,
        dropout_prob=0.3,
        wordvectors_path=wordvectors_path,
        vocab_path=vocab_path,
        freeze_embeddings=True
    )
    model.to(device)

    train_with_validation(model, train_dataloader, val_dataloader, epochs=50, lr=0.001, device=device)

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'models', 'intent_model_fast.pth'))
