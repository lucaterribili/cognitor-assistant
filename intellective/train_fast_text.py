import fasttext
import os
from config import BASE_DIR

def train_embedder():

    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

    corpus_path = os.path.join(BASE_DIR, 'data', 'fast-text.txt')
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(BASE_DIR, 'training_data', 'embeddings.txt')

    model = fasttext.train_unsupervised(
        input=corpus_path,
        model='skipgram',
        dim=300,
        epoch=25,  # Aumento le epoche per miglior apprendimento
        lr=0.1,
        minCount=1,  # Include parole rare
        wordNgrams=2,  # Usa character n-grams (3-6 caratteri di default)
        minn=2,  # Minimo n-gram size (per parole brevi come "ciao")
        maxn=5,  # Massimo n-gram size
        ws=5  # Context window size
    )


    model.save_model(MODEL_PATH)
    print(f"Modello FastText salvato in: {MODEL_PATH}")