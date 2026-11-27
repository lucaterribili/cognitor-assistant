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

from classes.intent_factorizer import IntentFactorizer
from classes.ner_tag_builder import NERTagBuilder
from config import BASE_DIR, USE_FASTTEXT, FACTORED_INTENTS
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


LABEL_SMOOTHING = 0.1  # evita che la loss di validazione cresca per eccesso di sicurezza mentre l'accuratezza sale


def intent_loss(model, criterion, intent_logits, topic_logits, type_logits, intent_labels):
    """
    Loss dell'intento: una CE sull'intento, oppure (modello a due teste) CE sull'argomento + CE sul tipo.
    Il tipo si addestra solo sugli intenti per cui ha senso (non sui saluti, conferme...).
    """
    if topic_logits is None:
        return criterion(intent_logits, intent_labels)
    loss = criterion(topic_logits, model.pair_topic[intent_labels])
    applies = model.type_applies[intent_labels].bool()
    if applies.any():
        loss = loss + criterion(type_logits[applies], model.pair_type[intent_labels][applies])
    return loss


def expected_calibration_error(confidences, correct, bins: int = 15) -> float:
    """ECE: quanto la confidenza dichiarata si discosta dall'accuratezza reale (0 = perfettamente calibrato)."""
    confidences, correct = torch.cat(confidences), torch.cat(correct).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(correct[in_bin].mean().item() - confidences[in_bin].mean().item())
    return ece


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device) -> dict:
    """
    Accuratezza sul dataloader: `intent` (intento completo) e, per ogni modello, `topic` e `type` ricavati
    dall'intento predetto (confrontabili tra modello a classe unica e a due teste). Il modello deve avere i buffer
    pair_topic / pair_type (vedi attach_factor_buffers) per calcolare topic e type.
    """
    model.eval()
    total = correct_intent = correct_topic = correct_type = 0
    confidences, corrects = [], []
    for inputs, intent_labels, ner_tags, masks in dataloader:
        inputs, intent_labels = inputs.to(device), intent_labels.to(device)
        intent_logits, _ = model(inputs)
        probs = torch.softmax(intent_logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        confidences.append(confidence)
        corrects.append(predicted == intent_labels)
        total += len(intent_labels)
        correct_intent += (predicted == intent_labels).sum().item()
        if hasattr(model, "pair_topic"):
            correct_topic += (model.pair_topic[predicted] == model.pair_topic[intent_labels]).sum().item()
            correct_type += (model.pair_type[predicted] == model.pair_type[intent_labels]).sum().item()
    if not total:
        return {}
    result = {"intent": correct_intent / total, "n": total, "ece": expected_calibration_error(confidences, corrects)}
    if hasattr(model, "pair_topic"):
        result.update(topic=correct_topic / total, type=correct_type / total)
    return result


def train_with_validation(model, train_dataloader, val_dataloader, epochs, lr, device, intent_weight=1.0, ner_weight=0.5, patience=10):
    intent_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_accuracy = -1.0
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
            intent_logits, ner_loss, topic_logits, type_logits = model(inputs, ner_tags=ner_tags, mask=masks, return_heads=True)
            loss_intent = intent_loss(model, intent_criterion, intent_logits, topic_logits, type_logits, intent_labels)
            loss = intent_weight * loss_intent + ner_weight * ner_loss
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
                intent_logits, ner_loss, topic_logits, type_logits = model(inputs, ner_tags=ner_tags, mask=masks, return_heads=True)
                loss_intent = intent_loss(model, intent_criterion, intent_logits, topic_logits, type_logits, intent_labels)
                loss = intent_weight * loss_intent + ner_weight * ner_loss
                total_val_loss += loss.item()
                total_val_intent_loss += loss_intent.item()
                total_val_ner_loss += ner_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_intent_loss = total_val_intent_loss / len(val_dataloader)
        avg_val_ner_loss = total_val_ner_loss / len(val_dataloader)

        scheduler.step(avg_val_loss)
        accuracy = evaluate_accuracy(model, val_dataloader, device)
        accuracy_text = ", ".join(f"{k}={v:.3f}" for k, v in accuracy.items() if k != "n")
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} "
              f"(Intent: {avg_val_intent_loss:.4f}, NER: {avg_val_ner_loss:.4f}) | Val acc: {accuracy_text}")

        # Early stopping e scelta del checkpoint sull'accuratezza di validazione (a parità, sulla loss):
        # la loss può salire per eccesso di sicurezza mentre l'accuratezza continua a migliorare.
        val_accuracy = accuracy.get("intent", 0.0)
        improved = val_accuracy > best_val_accuracy + 1e-4 or (
            abs(val_accuracy - best_val_accuracy) <= 1e-4 and avg_val_loss < best_val_loss)
        if improved:
            best_val_loss = avg_val_loss
            best_val_accuracy = max(best_val_accuracy, val_accuracy)
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"✓ Miglioramento! Val acc intento: {val_accuracy:.4f} (loss {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Pazienza: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping dopo {epoch + 1} epoche")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✓ Ripristinato il miglior modello (val acc intento {best_val_accuracy:.4f})")


def make_split(full_dataset):
    """Split train/validation group-aware (deterministico: random_state=42). Restituisce (train_idx, val_idx, info)."""
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
    info = {'groups': len(set(group_ids)), 'single_word_groups': len(single_word_groups)}
    return train_idx, val_idx, info


def train_main_model():
    device = torch.device("cpu")
    print(f"Dispositivo in uso: {device}")

    npy_path = os.path.join(BASE_DIR, '.cognitor', 'tokenized_data.npy')
    intent_dict_path = os.path.join(BASE_DIR, '.cognitor', 'intent_dict.json')

    with open(intent_dict_path, 'r') as f:
        intent_dict = json.load(f)
        intents_number = len(intent_dict)

    # Scomposizione degli intenti in (argomento, tipo di entità): due teste invece di una classe per intento.
    factors = None
    if FACTORED_INTENTS:
        factors = IntentFactorizer.from_intent_dict(intent_dict, NERTagBuilder().ENTITY_TYPES)
        factors.save(os.path.join(BASE_DIR, '.cognitor', 'intent_factors.json'))
        print(f"Intenti fattorizzati: {factors.num_topics} argomenti x {factors.num_types} tipi "
              f"-> {len(intent_dict)} intenti")

    full_dataset = IntentDataset(npy_path)

    train_idx, val_idx, split_info = make_split(full_dataset)

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx.tolist())
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx.tolist())
    print(f"Split group-aware: {len(train_dataset)} training, {len(val_dataset)} validation "
          f"({split_info['groups']} gruppi unici, {split_info['single_word_groups']} forzati in training "
          f"perché parola singola)")

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Carica vocab_size da vocab.json invece di FastText
    vocab_path = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    # Sia con FastText che con il nostro skip-gram (train_own_embedder), lo
    # step 2 della pipeline ha già scritto wordvectors.vec: l'embedding è
    # pre-addestrato e resta congelato durante il training del classifier,
    # indipendentemente da config.USE_FASTTEXT. Con USE_FASTTEXT=False il
    # tokenizer è invece a sotto-parola (SentencePiece) - vedi
    # classes/dataset_generator.py, che ha già tokenizzato tokenized_data.npy
    # con lo stesso modello.
    wordvectors_path = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')
    sentencepiece_model_path = None if USE_FASTTEXT else os.path.join(BASE_DIR, '.cognitor', 'sentencepiece.model')

    model = IntentClassifier(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=256,
        output_dim=intents_number,
        dropout_prob=0.3,
        wordvectors_path=wordvectors_path,
        vocab_path=vocab_path,
        freeze_embeddings=True,
        sentencepiece_model_path=sentencepiece_model_path,
        intent_factors=factors,
    )
    model.to(device)

    train_with_validation(model, train_dataloader, val_dataloader, epochs=50, lr=0.001, device=device)

    checkpoint = os.path.join(BASE_DIR, 'models', 'intent_model_fast.pth')
    torch.save(model.state_dict(), checkpoint)

    # Il label smoothing rende il modello sotto-sicuro: una temperatura sulla validation ricalibra le confidenze.
    from intellective.calibrate_intent_classifier import calibrate
    calibrate(checkpoint)


if __name__ == "__main__":
    train_main_model()
