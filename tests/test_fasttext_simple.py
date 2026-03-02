"""
Test veloce per confermare se FastText è il problema
"""
import os
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = '/home/luca/PycharmProjects/arianna-assistant'
fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

print("="*80)
print("TEST FASTTEXT: È LUI IL PROBLEMA?")
print("="*80)

ft_model = fasttext.load_model(fasttext_model_path)

# Test critico: similarità tra parole di categorie diverse
test_pairs = [
    ("ciao", "numero", False),    # saluto vs query
    ("ok", "capitale", False),    # conferma vs entità
    ("stop", "meteo", False),     # comando vs query
    ("aiuto", "prezzo", False),   # richiesta vs query
    ("ciao", "salve", True),      # entrambi saluti
    ("apri", "chiudi", True),     # entrambi comandi
]

print(f"\n🔍 Similarità coseno (ideale: simili >0.7, diversi <0.3):\n")

problems = 0
for w1, w2, should_be_similar in test_pairs:
    vec1 = ft_model.get_word_vector(w1)
    vec2 = ft_model.get_word_vector(w2)

    # Normalizza
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    sim = cosine_similarity([vec1_norm], [vec2_norm])[0][0]

    status = ""
    if should_be_similar and sim < 0.5:
        status = "❌ Troppo diversi!"
        problems += 1
    elif not should_be_similar and sim > 0.5:
        status = "❌ Troppo simili!"
        problems += 1
    else:
        status = "✓ OK"

    expected = "simili" if should_be_similar else "diversi"
    print(f"   '{w1:10s}' <-> '{w2:10s}': {sim:.4f} (dovrebbero essere {expected:7s}) {status}")

print(f"\n{'='*80}")
if problems >= 3:
    print("❌ PROBLEMA CONFERMATO: FastText NON discrimina bene!")
    print(f"   {problems} coppie problematiche trovate")
    print(f"\n💡 CAUSA: Corpus di training troppo piccolo (2753 linee)")
    print(f"   FastText ha bisogno di molto più testo per imparare")
    print(f"\n🔧 SOLUZIONE IMMEDIATA:")
    print(f"   1. Scarica embeddings pre-trainati per italiano")
    print(f"   2. Oppure aggiungi ~100k frasi al corpus e ri-traila")
    print(f"   3. Oppure usa una lookup table per parole corte comuni")
else:
    print("✓ FastText sembra OK, il problema è probabilmente architetturale")
    print(f"\n💡 CAUSA PROBABILE: GRU/Attention non adatti a sequenze di 1 token")

print("="*80)

