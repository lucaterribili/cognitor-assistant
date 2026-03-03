"""
Test diretto di FastText per vedere se genera embedding utili per sequenze corte
"""
import os
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace('/tests', '')
fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

print("="*80)
print("TEST FASTTEXT - EMBEDDING PER SEQUENZE CORTE")
print("="*80)

ft_model = fasttext.load_model(fasttext_model_path)

print(f"\n📊 Info modello:")
print(f"   • Dimensione: {ft_model.get_dimension()}")
print(f"   • Vocabolario: {len(ft_model.words)} parole")

# Test 1: Embeddings di parole corte comuni
print(f"\n{'='*80}")
print("TEST 1: QUALITÀ EMBEDDINGS PAROLE CORTE")
print("="*80)

short_words = [
    # Comandi base
    "ciao", "ok", "sì", "no", "stop", "aiuto", "grazie", "prego",
    # Verbi comuni
    "vai", "apri", "chiudi", "cerca", "trova", "dimmi",
    # Altre parole corte
    "che", "dove", "come", "quando", "chi", "cosa"
]

print(f"\nAnalisi embeddings:")
embeddings_dict = {}
for word in short_words:
    vec = ft_model.get_word_vector(word)
    embeddings_dict[word] = vec
    norm = np.linalg.norm(vec)
    mean = np.mean(vec)
    std = np.std(vec)
    print(f"   '{word:10s}': norm={norm:.4f}, mean={mean:.6f}, std={std:.4f}")

# Test 2: Similarità tra parole corte
print(f"\n{'='*80}")
print("TEST 2: SIMILARITÀ TRA PAROLE CORTE")
print("="*80)

# Parole simili semanticamente
similar_pairs = [
    ("ciao", "salve"),
    ("sì", "ok"),
    ("no", "stop"),
    ("aiuto", "help"),
    ("apri", "chiudi"),
]

print(f"\nSimilarità tra parole simili:")
for w1, w2 in similar_pairs:
    vec1 = ft_model.get_word_vector(w1)
    vec2 = ft_model.get_word_vector(w2)
    sim = cosine_similarity([vec1], [vec2])[0][0]
    print(f"   '{w1}' <-> '{w2}': {sim:.4f}")

# Parole diverse semanticamente
different_pairs = [
    ("ciao", "numero"),
    ("ok", "capitale"),
    ("stop", "meteo"),
]

print(f"\nSimilarità tra parole diverse:")
for w1, w2 in different_pairs:
    vec1 = ft_model.get_word_vector(w1)
    vec2 = ft_model.get_word_vector(w2)
    sim = cosine_similarity([vec1], [vec2])[0][0]
    print(f"   '{w1}' <-> '{w2}': {sim:.4f}")

# Test 3: OOV (Out of Vocabulary) words
print(f"\n{'='*80}")
print("TEST 3: GESTIONE OOV (Out-of-Vocabulary)")
print("="*80)

oov_words = ["asdfgh", "xyzabc", "qqqqq"]
print(f"\nParole OOV (non nel vocabolario):")
for word in oov_words:
    vec = ft_model.get_word_vector(word)
    norm = np.linalg.norm(vec)
    print(f"   '{word}': norm={norm:.4f} (FastText usa subword)")

# Test 4: Sentence embedding medio
print(f"\n{'='*80}")
print("TEST 4: SENTENCE EMBEDDING (media dei token)")
print("="*80)

test_sentences = [
    "ciao",
    "ok",
    "aiuto",
    "che ore sono",
    "vorrei sapere che ore sono"
]

def get_sentence_embedding(text, ft_model):
    """Calcola embedding medio della frase"""
    tokens = text.lower().split()
    vecs = [ft_model.get_word_vector(t) for t in tokens]
    return np.mean(vecs, axis=0)

sentence_embeddings = {}
for sentence in test_sentences:
    vec = get_sentence_embedding(sentence, ft_model)
    sentence_embeddings[sentence] = vec
    norm = np.linalg.norm(vec)
    print(f"\n   '{sentence}'")
    print(f"      Tokens: {len(sentence.split())}")
    print(f"      Norm: {norm:.4f}")

# Similarità tra frasi corte e lunghe con stesso intent
print(f"\n\nSimilarità sentence embedding:")
time_queries = ["ore", "che ore sono", "vorrei sapere che ore sono"]
time_vecs = [get_sentence_embedding(s, ft_model) for s in time_queries]

for i in range(len(time_queries)):
    for j in range(i+1, len(time_queries)):
        sim = cosine_similarity([time_vecs[i]], [time_vecs[j]])[0][0]
        print(f"   '{time_queries[i]}' <-> '{time_queries[j]}'")
        print(f"      Similarità: {sim:.4f}")

print(f"\n{'='*80}")
print("CONCLUSIONI SUL FASTTEXT")
print("="*80)

# Calcola varianza delle norme per parole corte
norms = [np.linalg.norm(embeddings_dict[w]) for w in short_words]
avg_norm = np.mean(norms)
std_norm = np.std(norms)

print(f"\n📊 Statistiche embeddings parole corte:")
print(f"   • Norma media: {avg_norm:.4f}")
print(f"   • Std dev norme: {std_norm:.4f}")
print(f"   • Coefficiente variazione: {(std_norm/avg_norm)*100:.2f}%")

if std_norm < 1.0:
    print(f"\n⚠️  PROBLEMA FASTTEXT POSSIBILE:")
    print(f"   Le norme degli embedding sono molto simili tra loro.")
    print(f"   FastText potrebbe non discriminare bene parole corte diverse.")
    print(f"\n💡 SOLUZIONE:")
    print(f"   • Ri-trainare FastText con più dati")
    print(f"   • Oppure usare embeddings pre-trainati (word2vec, GloVe)")
    print(f"   • Oppure usare character-level embeddings per parole corte")
else:
    print(f"\n✓ FastText sembra generare embeddings sufficientemente diversi")
    print(f"\n💡 Il problema probabilmente NON è FastText, ma:")
    print(f"   • Architettura del modello (GRU+Attention con 1 token)")
    print(f"   • Distribuzione degli intents nel training set")
    print(f"   • Necessità di più esempi di training per sequenze corte")

print()

