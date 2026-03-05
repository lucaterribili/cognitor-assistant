"""
Verifica i dati di training per il NER
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import os
from config import BASE_DIR
from classes.ner_tag_builder import NERTagBuilder

def main():
    print("=" * 80)
    print("ANALISI DATI DI TRAINING NER")
    print("=" * 80)

    # Carica i dati tokenizzati
    npy_path = os.path.join(BASE_DIR, '.cognitor', 'tokenized_data.npy')
    data = np.load(npy_path, allow_pickle=True)

    # Carica NER tag builder
    tag_builder_path = os.path.join(BASE_DIR, '.cognitor', 'ner_tag_builder.json')
    if os.path.exists(tag_builder_path):
        ner_tag_builder = NERTagBuilder.load(tag_builder_path)
    else:
        ner_tag_builder = NERTagBuilder()

    print(f"\n1. Informazioni sul dataset:")
    print(f"   - Numero di esempi: {len(data)}")
    print(f"   - Numero di tag NER: {ner_tag_builder.num_tags}")

    # Analizza distribuzione tag NER
    tag_counts = {}
    examples_with_entities = 0

    for i, item in enumerate(data):
        token_ids = item[0]
        intent_id = item[1][0]
        ner_tag_ids = item[2] if len(item) > 2 else []

        if len(ner_tag_ids) == 0:
            print(f"   ⚠️  Esempio {i} non ha tag NER!")
            continue

        has_entity = False
        for tag_id in ner_tag_ids:
            tag_name = ner_tag_builder.id2tag.get(tag_id, "UNKNOWN")
            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
            if tag_id != 0:  # Non è O
                has_entity = True

        if has_entity:
            examples_with_entities += 1

    print(f"\n2. Statistiche NER:")
    print(f"   - Esempi con entità: {examples_with_entities}/{len(data)} ({examples_with_entities/len(data)*100:.1f}%)")
    print(f"\n3. Distribuzione tag:")

    # Ordina per frequenza
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags[:20]:  # Top 20
        print(f"   - {tag:15s}: {count:6d} ({count/sum(tag_counts.values())*100:5.2f}%)")

    # Mostra alcuni esempi con entità
    print(f"\n4. Esempi con entità (primi 10):")
    count = 0
    for i, item in enumerate(data):
        token_ids = item[0]
        ner_tag_ids = item[2] if len(item) > 2 else []

        # Controlla se ha entità
        has_entity = any(tag_id != 0 for tag_id in ner_tag_ids)

        if has_entity:
            count += 1
            print(f"\n   Esempio {i}:")
            print(f"   - Token IDs: {token_ids[:10]}...")
            print(f"   - NER tags: {[ner_tag_builder.id2tag[tid] for tid in ner_tag_ids]}")

            if count >= 10:
                break

    # Verifica se ci sono problemi di lunghezza
    print(f"\n5. Verifica consistenza lunghezze:")
    mismatches = 0
    for i, item in enumerate(data):
        token_ids = item[0]
        ner_tag_ids = item[2] if len(item) > 2 else []

        if len(token_ids) != len(ner_tag_ids):
            mismatches += 1
            if mismatches <= 5:  # Mostra i primi 5
                print(f"   ⚠️  Esempio {i}: {len(token_ids)} token vs {len(ner_tag_ids)} tag")

    if mismatches == 0:
        print(f"   ✓ Tutte le sequenze hanno lunghezze consistenti")
    else:
        print(f"   ❌ {mismatches} esempi con lunghezze inconsistenti!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

