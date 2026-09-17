"""
Addestra un embedding word-level (skip-gram + negative sampling, stile
word2vec) sul corpus del dataset, come alternativa a FastText.

Stesso ruolo di train_fast_text.train_embedder() (fase di pre-training
separata, prima del classifier) e stesso formato di output (.vec + vocab.json)
- IntentClassifier.load_word_vectors() carica l'uno o l'altro senza saperne
la provenienza.

Differenza rispetto a FastText: nessuna informazione di sottoparola. FastText
generalizza a parole mai viste nel corpus tramite n-grammi di carattere
condivisi con parole simili (es. "lasagne" ~ "lasagna"); questo embedding
word-level no - una parola assente dal corpus resta assente dal vocabolario,
il tokenizer la tratta come sconosciuta (vedi classes/simple_tokenizer.py).
Quello che guadagniamo rispetto a inizializzare l'embedding a caso e
addestrarlo solo con il segnale di classificazione: le co-occorrenze
osservate su TUTTO il corpus (non solo negli esempi etichettati) piazzano già
parole semanticamente vicine vicine tra loro, prima ancora che il classifier
inizi ad addestrarsi.
"""
import os
import re
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BASE_DIR

_PUNCT_PATTERN = re.compile(r'[^\w\s]')

_EMBED_DIM = 300
_WINDOW_SIZE = 5
_NEGATIVE_SAMPLES = 5
_EPOCHS = 25
_LR = 0.01
_BATCH_SIZE = 512


def _tokenize_corpus(corpus_path: str) -> list[list[str]]:
    """Stessa tokenizzazione di SimpleTokenizer: lowercase, punteggiatura -> spazio, split."""
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned = _PUNCT_PATTERN.sub(' ', line.strip().lower())
            tokens = cleaned.split()
            if tokens:
                sentences.append(tokens)
    return sentences


def _build_vocab(sentences: list[list[str]]) -> tuple[list[str], dict[str, int], list[int]]:
    """
    Vocabolario ordinato per frequenza decrescente (alfabetico a parità), come
    build_vocab_only() in train_fast_text.py - stesso ordine deterministico.
    """
    counts = Counter(word for sentence in sentences for word in sentence)
    words = [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    word_to_idx = {w: i for i, w in enumerate(words)}
    freqs = [counts[w] for w in words]
    return words, word_to_idx, freqs


def _build_skipgram_pairs(
    sentences: list[list[str]], word_to_idx: dict[str, int], window_size: int
) -> np.ndarray:
    """
    Genera coppie (centro, contesto) da tutte le frasi del corpus, con
    finestra dinamica (1..window_size per ogni parola centro, trucco standard
    di word2vec che pesa più le parole vicine di quelle ai margini).
    """
    pairs = []
    for sentence in sentences:
        indices = [word_to_idx[w] for w in sentence]
        n = len(indices)
        for i, center in enumerate(indices):
            window = random.randint(1, window_size)
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, indices[j]))
    return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


class _SkipGramModel(nn.Module):
    """Skip-gram con negative sampling: due matrici di embedding (centro/contesto)."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        # Init stile word2vec: pesi piccoli centrati sullo zero per il centro,
        # zero per il contesto (valore comune nelle implementazioni SGNS).
        nn.init.uniform_(self.in_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        v_center = self.in_embed(centers)  # (B, D)
        v_context = self.out_embed(contexts)  # (B, D)
        v_neg = self.out_embed(negatives)  # (B, K, D)

        pos_score = torch.sum(v_center * v_context, dim=1)  # (B,)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(v_neg, v_center.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


def train_own_embedder():
    """
    Allena l'embedder "nostro" (no FastText) sul corpus generato dalla
    pipeline. Stesso ruolo/output di train_fast_text.train_embedder():

    Salva:
    - .cognitor/wordvectors.vec: matrice statica parola -> vettore
    - .cognitor/vocab.json: vocabolario puro per il tokenizer
    """
    VEC_PATH = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')
    VOCAB_PATH = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')

    corpus_path = os.path.join(BASE_DIR, '.cognitor', 'fast-text.txt')
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Corpus non trovato! Esegui prima: python -m pipeline.intent_builder\n"
            f"Path cercati:\n"
            f"  - {os.path.join(BASE_DIR, '.cognitor', 'fast-text.txt')}\n"
            f"  - {os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')}"
        )

    print(f"Training embedder (skip-gram, no FastText) con corpus: {corpus_path}")

    sentences = _tokenize_corpus(corpus_path)
    words, word_to_idx, freqs = _build_vocab(sentences)
    vocab_size = len(words)
    print(f"Vocabolario: {vocab_size} parole, {len(sentences)} frasi")

    pairs = _build_skipgram_pairs(sentences, word_to_idx, _WINDOW_SIZE)
    print(f"Coppie skip-gram generate: {len(pairs)}")

    device = torch.device("cpu")
    model = _SkipGramModel(vocab_size, _EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)

    # Distribuzione di rumore per il negative sampling: frequenza^0.75, trucco
    # standard di word2vec per non sovra-pescare le poche parole più comuni.
    noise_dist = torch.tensor(freqs, dtype=torch.float64) ** 0.75
    noise_dist = noise_dist / noise_dist.sum()

    n_pairs = len(pairs)
    for epoch in range(_EPOCHS):
        perm = np.random.permutation(n_pairs)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, _BATCH_SIZE):
            batch = pairs[perm[start:start + _BATCH_SIZE]]
            centers = torch.from_numpy(batch[:, 0]).to(device)
            contexts = torch.from_numpy(batch[:, 1]).to(device)
            negatives = torch.multinomial(
                noise_dist, len(batch) * _NEGATIVE_SAMPLES, replacement=True
            ).view(len(batch), _NEGATIVE_SAMPLES).to(device)

            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{_EPOCHS}: loss={avg_loss:.4f}")

    # Le vettore finali sono la matrice "centro" (in_embed), come da
    # convenzione word2vec (quella usata più comunemente come embedding finale).
    embedding_matrix = model.in_embed.weight.detach().cpu().numpy()

    with open(VEC_PATH, 'w') as f:
        f.write(f"{vocab_size} {_EMBED_DIM}\n")
        for i, word in enumerate(words):
            vec = embedding_matrix[i]
            f.write(word + " " + " ".join(str(x) for x in vec) + "\n")

    print(f"✓ Word vectors salvati: {VEC_PATH}")

    with open(VOCAB_PATH, 'w') as f:
        json.dump(words, f)

    print(f"✓ Vocabolario salvato: {VOCAB_PATH}")
    print(f"\nTraining embedder (no FastText) completato. {vocab_size} parole, {_EMBED_DIM} dimensioni.")
