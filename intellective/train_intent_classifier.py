import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import GroupShuffleSplit
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

    def get_group_ids(self) -> list:
        """
        Restituisce il group_id di ogni riga (esempio sorgente originale prima
        di normalizzazione/doping). Usato per fare uno split train/val
        group-aware ed evitare che varianti near-duplicate della stessa frase
        finiscano su lati opposti dello split (data leakage).

        Se il .npy e' stato generato prima dell'introduzione del group_id
        (nessun 4° campo), ogni riga riceve un group_id univoco: si ricade
        semplicemente nel comportamento precedente (split a livello di riga).
        """
        group_ids = []
        for idx in range(len(self.data)):
            item = self.data[idx]
            group_id = item[3] if len(item) > 3 and item[3] else f"__row_{idx}"
            group_ids.append(group_id)
        return group_ids

    def get_token_lengths(self) -> list:
        """
        Numero di token per ogni riga. Il tokenizer di questo progetto (vedi
        SimpleTokenizer) splitta sugli spazi, quindi lunghezza 1 equivale a
        un esempio di una sola parola (es. "ciao"). Usato per forzare questi
        esempi-prototipo nel training set (vedi train_main_model): sono gli
        esempi più rappresentativi di un intent e perderli per caso nello
        split di validation lascia il modello incapace di classificarli.
        """
        return [len(self.data[idx][0]) for idx in range(len(self.data))]


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
    device = torch.device("cpu")
    print(f"Dispositivo in uso: {device}")

    npy_path = os.path.join(BASE_DIR, '.cognitor', 'tokenized_data.npy')
    intent_dict_path = os.path.join(BASE_DIR, '.cognitor', 'intent_dict.json')

    with open(intent_dict_path, 'r') as f:
        intent_dict = json.load(f)
        intents_number = len(intent_dict)

    full_dataset = IntentDataset(npy_path)

    # Split train/validation (80/20) GROUP-AWARE: tutte le varianti
    # (clean_text/normalizzato/doping) dello stesso esempio sorgente
    # condividono un group_id e devono finire sempre dalla stessa parte
    # dello split, altrimenti la validation accuracy e' gonfiata perche'
    # il modello ha gia' visto in training una quasi-copia dell'esempio
    # su cui viene "validato".
    #
    # I gruppi di UNA sola parola (es. "ciao") sono forzati in training e
    # esclusi dal sorteggio: sono gli esempi-prototipo più rappresentativi
    # di un intent, e lasciarli finire per caso in validation significa
    # rischiare che il modello non impari mai a classificarli con sicurezza
    # pur essendo l'esempio più semplice e comune per quell'intent.
    group_ids = full_dataset.get_group_ids()
    token_lengths = full_dataset.get_token_lengths()

    single_word_groups = set()
    for group_id, length in zip(group_ids, token_lengths):
        if length <= 1:
            single_word_groups.add(group_id)

    splittable_mask = [group_id not in single_word_groups for group_id in group_ids]
    splittable_indices = np.array([i for i, keep in enumerate(splittable_mask) if keep])
    forced_train_indices = np.array([i for i, keep in enumerate(splittable_mask) if not keep])

    unique_splittable_groups = {group_ids[i] for i in splittable_indices}
    if len(splittable_indices) > 0 and len(unique_splittable_groups) >= 2:
        splittable_group_ids = [group_ids[i] for i in splittable_indices]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        rel_train_idx, rel_val_idx = next(
            splitter.split(np.zeros(len(splittable_group_ids)), groups=splittable_group_ids)
        )
        split_train_idx = splittable_indices[rel_train_idx]
        val_idx = splittable_indices[rel_val_idx]
    else:
        # Troppo pochi gruppi "splittabili" per uno split sensato: tutto in training.
        split_train_idx = splittable_indices
        val_idx = np.array([], dtype=int)

    train_idx = np.concatenate([forced_train_indices, split_train_idx])

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx.tolist())
    print(f"Split group-aware: {len(train_dataset)} training, {len(val_dataset)} validation "
          f"({len(set(group_ids))} gruppi unici, {len(single_word_groups)} forzati in training "
          f"perché parola singola)")

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
