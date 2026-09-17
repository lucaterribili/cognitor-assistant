"""
Addestra un embedding di sotto-parola (skip-gram + negative sampling, stile
word2vec, sopra un vocabolario SentencePiece/BPE) sul corpus del dataset,
come alternativa a FastText.

Skip-gram scritto a mano in PyTorch (Adam, learning rate costante/adattivo
per parametro) invece di gensim.models.Word2Vec (SGD con decadimento lineare
del learning rate, pensato per corpora enormi): a parità di modello
SentencePiece e di dataset per il classifier, gensim ha dato una val loss
del classifier misurabilmente peggiore (1.99 contro 1.81, con "come si
preparano le lasagne" che finiva out_of_scope invece di ask_recipe) - sul
nostro corpus piccolo (~90k token) lo schedule di learning rate pensato per
milioni di parole non converge bene nel numero di epoche che usiamo qui.
Tornati a questa versione dopo il confronto diretto.

Stesso ruolo di train_fast_text.train_embedder() (fase di pre-training
separata, prima del classifier) e stesso formato di output (.vec + vocab.json)
- IntentClassifier.load_word_vectors() carica l'uno o l'altro senza saperne
la provenienza. Il vocabolario non è fatto di parole intere ma di
sotto-parole: qualsiasi stringa - anche mai vista nel training - si
scompone sempre in pezzi noti (fino al singolo byte come fallback estremo),
invece di risultare "fuori vocabolario". Vedi
classes/sentencepiece_tokenizer.py per il dettaglio del meccanismo e
classes/dataset_generator.py per dove viene consumato.
"""
import os
import json
import random

import sentencepiece as spm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BASE_DIR

_SP_VOCAB_SIZE = 2000
_EMBED_DIM = 300
_WINDOW_SIZE = 5
_NEGATIVE_SAMPLES = 5
_EPOCHS = 25
_LR = 0.01
_BATCH_SIZE = 512


def _train_sentencepiece(corpus_path: str) -> spm.SentencePieceProcessor:
    """
    Allena il modello SentencePiece (BPE) sul corpus e lo salva in
    .cognitor/sentencepiece.model - stesso path che
    classes/sentencepiece_tokenizer.py e agent/model_loader.py si aspettano.

    Il corpus viene lowercased in un file temporaneo prima del training: resta
    così coerente con il resto della pipeline, che lowercasa sempre il testo
    prima di tokenizzare (vedi SentencePieceTokenizer.tokenize_with_ids).
    """
    model_prefix = os.path.join(BASE_DIR, '.cognitor', 'sentencepiece')
    lower_corpus_path = os.path.join(BASE_DIR, '.cognitor', 'fast-text-lower.txt')

    with open(corpus_path, 'r', encoding='utf-8') as src, \
            open(lower_corpus_path, 'w', encoding='utf-8') as dst:
        for line in src:
            dst.write(line.lower())

    spm.SentencePieceTrainer.train(
        input=lower_corpus_path,
        model_prefix=model_prefix,
        vocab_size=_SP_VOCAB_SIZE,
        model_type='bpe',
        character_coverage=1.0,
        # Qualsiasi carattere/simbolo mai visto nel training (emoji, alfabeti
        # esteri, ...) si scompone in byte invece di diventare <unk>: nessun
        # input risulta mai completamente fuori vocabolario.
        byte_fallback=True,
    )

    print(f"✓ Modello SentencePiece salvato: {model_prefix}.model")
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


def _encode_corpus(sp: spm.SentencePieceProcessor, corpus_path: str) -> list[list[int]]:
    sequences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                sequences.append(sp.encode(line, out_type=int))
    return sequences


def _build_skipgram_pairs(sequences: list[list[int]], window_size: int) -> np.ndarray:
    """
    Genera coppie (centro, contesto) da tutte le sequenze del corpus, con
    finestra dinamica (1..window_size per ogni id centro, trucco standard di
    word2vec che pesa più i vicini immediati di quelli ai margini).
    """
    pairs = []
    for ids in sequences:
        n = len(ids)
        for i, center in enumerate(ids):
            window = random.randint(1, window_size)
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, ids[j]))
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
    - .cognitor/sentencepiece.model: modello di tokenizzazione a sotto-parola
    - .cognitor/wordvectors.vec: matrice statica pezzo -> vettore
    - .cognitor/vocab.json: vocabolario puro (i pezzi SentencePiece, in ordine
      di id) per il tokenizer
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

    print(f"Training embedder (skip-gram su sotto-parole SentencePiece, no FastText) con corpus: {corpus_path}")

    sp = _train_sentencepiece(corpus_path)
    vocab_size = sp.vocab_size()
    words = [sp.id_to_piece(i) for i in range(vocab_size)]

    sequences = _encode_corpus(sp, corpus_path)
    print(f"Vocabolario: {vocab_size} pezzi, {len(sequences)} frasi")

    pairs = _build_skipgram_pairs(sequences, _WINDOW_SIZE)
    print(f"Coppie skip-gram generate: {len(pairs)}")

    device = torch.device("cpu")
    model = _SkipGramModel(vocab_size, _EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)

    # Distribuzione di rumore per il negative sampling: frequenza^0.75, trucco
    # standard di word2vec per non sovra-pescare i pochi pezzi più comuni.
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for ids in sequences:
        for token_id in ids:
            freqs[token_id] += 1
    freqs = np.maximum(freqs, 1.0)  # evita peso zero per pezzi mai centro/contesto
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

    # I vettori finali sono la matrice "centro" (in_embed), come da convenzione
    # word2vec (quella usata più comunemente come embedding finale).
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
    print(f"\nTraining embedder (no FastText) completato. {vocab_size} pezzi, {_EMBED_DIM} dimensioni.")
