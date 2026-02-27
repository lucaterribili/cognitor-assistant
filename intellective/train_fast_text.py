import fasttext
import os
from config import BASE_DIR

def train_embedder():

    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

    corpus_path = os.path.join(BASE_DIR, 'data', 'fast-text.txt')
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(BASE_DIR, 'training_data', 'fasttext_phrases.txt')

    model = fasttext.train_unsupervised(
        input=corpus_path,
        model='skipgram',
        dim=300,
        epoch=20,
        lr=0.1,
        minCount=1,  # <- abbassa a 1, i subword sono rari
        wordNgrams=1  # <- abbassa a 1, i subword sono già n-gram
    )


    model.save_model(MODEL_PATH)
    print(f"Modello FastText salvato in: {MODEL_PATH}")