import fasttext
import os
from config import BASE_DIR

def train_embedder():
    """
    Allena il modello FastText per gli embeddings.
    Usa il corpus generato da DatasetGenerator (data/fast-text.txt).
    """
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

    # Usa il corpus generato dalla pipeline
    corpus_path = os.path.join(BASE_DIR, 'data', 'fast-text.txt')

    # Fallback se non esiste (legacy)
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Corpus non trovato! Esegui prima: python -m pipeline.intent_builder\n"
            f"Path cercati:\n"
            f"  - {os.path.join(BASE_DIR, 'data', 'fast-text.txt')}\n"
            f"  - {os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')}"
        )

    print(f"Training FastText con corpus: {corpus_path}")

    model = fasttext.train_unsupervised(
        input=corpus_path,
        model='skipgram',
        dim=300,
        epoch=25,
        lr=0.1,
        minCount=1,
        wordNgrams=2,
        minn=2,
        maxn=5,
        ws=5
    )


    model.save_model(MODEL_PATH)
    print(f"Modello FastText salvato in: {MODEL_PATH}")