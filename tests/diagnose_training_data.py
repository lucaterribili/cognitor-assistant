"""
Analisi dei dati di training per capire la distribuzione delle sequenze corte
"""
import json
import os
import numpy as np
from collections import defaultdict
from config import BASE_DIR

print("="*80)
print("ANALISI DATI DI TRAINING - DISTRIBUZIONE LUNGHEZZE")
print("="*80)

# Carica dati tokenizzati
npy_path = os.path.join(BASE_DIR, '.cognitor', 'tokenized_data.npy')
intent_dict_path = os.path.join(BASE_DIR, '.cognitor', 'intent_dict.json')

data = np.load(npy_path, allow_pickle=True)
with open(intent_dict_path, 'r') as f:
    intent_dict = json.load(f)

# Crea reverse mapping
id_to_intent = {int(k): v for k, v in intent_dict.items()}

print(f"\n📊 Dataset:")
print(f"   • Totale esempi: {len(data)}")
print(f"   • Numero intents: {len(intent_dict)}")

# Analizza lunghezze
length_distribution = defaultdict(int)
length_by_intent = defaultdict(lambda: defaultdict(int))
examples_by_length = defaultdict(list)

for item in data:
    tokens = item[0]
    intent_id = item[1][0]
    intent_name = id_to_intent[intent_id]
    seq_len = len(tokens)

    length_distribution[seq_len] += 1
    length_by_intent[seq_len][intent_name] += 1
    examples_by_length[seq_len].append({
        'tokens': tokens,
        'intent': intent_name,
        'intent_id': intent_id
    })

print(f"\n{'='*80}")
print("DISTRIBUZIONE LUNGHEZZE SEQUENZE NEL TRAINING SET")
print("="*80)

for seq_len in sorted(length_distribution.keys()):
    count = length_distribution[seq_len]
    percentage = (count / len(data)) * 100
    print(f"\n📏 Lunghezza {seq_len} token(s): {count} esempi ({percentage:.2f}%)")

    # Mostra intents per questa lunghezza
    intents_for_len = length_by_intent[seq_len]
    sorted_intents = sorted(intents_for_len.items(), key=lambda x: x[1], reverse=True)

    print(f"   Top intents:")
    for intent_name, intent_count in sorted_intents[:5]:
        print(f"      • {intent_name}: {intent_count} esempi")

# Focus su sequenze molto corte
print(f"\n{'='*80}")
print("FOCUS: SEQUENZE DI 1 TOKEN")
print("="*80)

if 1 in examples_by_length:
    one_token_examples = examples_by_length[1]
    print(f"\n📝 Trovati {len(one_token_examples)} esempi di 1 token")
    print(f"\nEsempi:")

    # Raggruppa per intent
    by_intent = defaultdict(list)
    for ex in one_token_examples:
        by_intent[ex['intent']].append(ex['tokens'][0])

    for intent_name in sorted(by_intent.keys()):
        token_ids = by_intent[intent_name][:10]  # Max 10 esempi
        print(f"\n   Intent: {intent_name}")
        print(f"   Token IDs: {token_ids}")
else:
    print(f"\n⚠️  Nessun esempio di 1 token nel training set!")
    print(f"   QUESTO È IL PROBLEMA! Il modello non ha mai visto sequenze di 1 token.")

print(f"\n{'='*80}")
print("FOCUS: SEQUENZE DI 2 TOKEN")
print("="*80)

if 2 in examples_by_length:
    two_token_examples = examples_by_length[2]
    print(f"\n📝 Trovati {len(two_token_examples)} esempi di 2 token")
    print(f"\nEsempi (primi 10):")

    for i, ex in enumerate(two_token_examples[:10]):
        print(f"   {i+1}. Token IDs: {ex['tokens']} -> Intent: {ex['intent']}")
else:
    print(f"\n⚠️  Nessun esempio di 2 token nel training set!")

# Statistiche generali
print(f"\n{'='*80}")
print("STATISTICHE GENERALI")
print("="*80)

lengths = [len(item[0]) for item in data]
print(f"\n📊 Lunghezza sequenze:")
print(f"   • Media: {np.mean(lengths):.2f} token")
print(f"   • Mediana: {np.median(lengths):.0f} token")
print(f"   • Min: {np.min(lengths)} token")
print(f"   • Max: {np.max(lengths)} token")
print(f"   • Std dev: {np.std(lengths):.2f}")

# Percentili
print(f"\n   Percentili:")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"      {p}°: {np.percentile(lengths, p):.0f} token")

# Quante sono "corte" (1-3 token)?
short_count = sum(1 for l in lengths if l <= 3)
short_percentage = (short_count / len(lengths)) * 100
print(f"\n   📉 Sequenze ≤3 token: {short_count}/{len(lengths)} ({short_percentage:.2f}%)")

medium_count = sum(1 for l in lengths if 4 <= l <= 7)
medium_percentage = (medium_count / len(lengths)) * 100
print(f"   📊 Sequenze 4-7 token: {medium_count}/{len(lengths)} ({medium_percentage:.2f}%)")

long_count = sum(1 for l in lengths if l >= 8)
long_percentage = (long_count / len(lengths)) * 100
print(f"   📈 Sequenze ≥8 token: {long_count}/{len(lengths)} ({long_percentage:.2f}%)")

print(f"\n{'='*80}")
print("DIAGNOSI FINALE")
print("="*80)

if short_percentage < 10:
    print(f"\n❌ PROBLEMA TROVATO: Solo {short_percentage:.2f}% di esempi corti nel training!")
    print(f"   Il modello NON ha imparato a gestire sequenze corte.")
    print(f"\n💡 SOLUZIONI:")
    print(f"   1. Data Augmentation: aggiungi più esempi di 1-3 token")
    print(f"   2. FastText è OK, il problema è il training set sbilanciato")
    print(f"   3. Considera di rimuovere/semplificare il GRU per seq corte")
elif short_percentage > 30:
    print(f"\n✓ Buona rappresentazione di sequenze corte: {short_percentage:.2f}%")
    print(f"\n💡 Se il problema persiste, potrebbe essere:")
    print(f"   1. FastText genera embeddings troppo simili per parole corte")
    print(f"   2. Il GRU non è adatto per sequenze di 1 token")
    print(f"   3. Attention non funziona bene con 1 solo token")
else:
    print(f"\n⚠️  Rappresentazione moderata di sequenze corte: {short_percentage:.2f}%")
    print(f"   Potrebbe essere migliorata con più data augmentation")

print(f"\n")

