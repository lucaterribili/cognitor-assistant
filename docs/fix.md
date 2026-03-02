Sembra che il filesystem non sia disponibile in questa sessione. Ti incollo direttamente il markdown qui — puoi copiarlo e salvarlo tu come `modifiche.md`.

---

# Modifiche da apportare

## 1. `simple_tokenizer.py`

```python
class SimpleTokenizer:
    def __init__(self, fasttext_model_path=None):
        self.pattern = re.compile(r'[^\w\s]')
        self.fasttext_model = None
        self.vocab = None
        self.word_to_idx = {}
        self.pad_id = 0  # ← AGGIUNTO

        if fasttext_model_path and os.path.exists(fasttext_model_path):
            import fasttext
            self.fasttext_model = fasttext.load_model(fasttext_model_path)
            self.vocab = self.fasttext_model.words
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

    def get_word_index(self, word):
        if self.fasttext_model and word in self.word_to_idx:
            return self.word_to_idx[word]
        return 1  # ← MODIFICATO: OOV → 1, padding rimane 0
```

---

## 2. `intent_classifier.py` — `forward`

```python
def forward(self, x, ner_tags=None, mask=None):
    batch_size, seq_len = x.shape

    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)

    x_embedded = self.embedding(x)
    lengths = mask.sum(1)
    is_short = lengths < 3

    gru_out, _ = self.bigru(x_embedded)

    attn_out = self.attention(gru_out)
    intent_logits = self.fc(self.dropout(attn_out))

    if is_short.any():
        float_mask = mask.unsqueeze(-1).float()
        short_embed = (x_embedded * float_mask).sum(1) / float_mask.sum(1).clamp(min=1)
        short_intent_logits = self.short_seq_fc(self.short_seq_dropout(short_embed))
        intent_logits = torch.where(is_short.unsqueeze(1), short_intent_logits, intent_logits)

    ner_emissions = self.ner_fc(self.ner_dropout(gru_out))

    if is_short.any():
        float_mask = mask.unsqueeze(-1).float()
        short_embed = (x_embedded * float_mask).sum(1) / float_mask.sum(1).clamp(min=1)
        short_ner_feat = self.short_seq_ner_proj(short_embed)
        short_ner_emissions = self.ner_fc(short_ner_feat.unsqueeze(1).expand(-1, seq_len, -1))
        ner_emissions = torch.where(is_short.view(batch_size, 1, 1), short_ner_emissions, ner_emissions)

    if ner_tags is not None:
        ner_loss = -self.crf(ner_emissions, ner_tags, mask=mask, reduction='mean')
        return intent_logits, ner_loss
    else:
        ner_predictions = self.crf.decode(ner_emissions, mask=mask)
        return intent_logits, ner_predictions
```

---

## 3. `train_intent_classifier.py` — `train_model`

```python
def train_model(model, dataloader, epochs, lr, device, intent_weight=1.0, ner_weight=0.5, patience=10, class_weights=None):

    if class_weights is not None:
        intent_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        intent_criterion = nn.CrossEntropyLoss()
    
    # tutto il resto invariato
```

---

## 4. `train_intent_classifier.py` — `train_main_model`

```python
    dataset = IntentDataset(npy_path)

    # Verifica distribuzione lunghezze
    lengths = [len(dataset[i][0]) for i in range(len(dataset))]
    short = sum(1 for l in lengths if l < 3)
    print(f"Sequenze corte (<3 token): {short}/{len(dataset)} ({100*short/len(dataset):.1f}%)")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Weighted loss
    from collections import Counter
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    weights = torch.zeros(intents_number)
    for idx, count in label_counts.items():
        weights[idx] = total / (intents_number * count)

    # ... definizione modello invariata ...

    train_model(model, dataloader, epochs=50, lr=0.001, device=device, class_weights=weights)
```

---

## Ordine di esecuzione

1. Modifica `simple_tokenizer.py`
2. Modifica `forward` in `intent_classifier.py`
3. Modifica `train_model` e `train_main_model` in `train_intent_classifier.py`
4. Avvia il training e controlla il log delle sequenze corte — se sono sotto il 5% del totale, aggiungi esempi brevi al dataset prima di procedere
5. Riavvia il training da zero