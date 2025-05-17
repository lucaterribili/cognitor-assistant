import fasttext
import os
from config import BASE_DIR

# Modello Skip-gram (alternativa: CBOW con model='cbow')
model = fasttext.train_unsupervised(
    input=os.path.join(BASE_DIR, 'data', 'fast-text.txt'),
    model='skipgram',    # 'skipgram' o 'cbow'
    dim=300,             # Dimensione degli embedding
    epoch=20,            # Numero di epoche
    lr=0.1,             # Learning rate
    minCount=3,          # Ignora parole con meno di 3 occorrenze
    wordNgrams=2         # Considera bigrammi
)

# Salva il modello
model.save_model("fasttext_model.bin")