"""
Analisi delle annotazioni NER nei dati
"""

import json
import os
from collections import Counter
from config import BASE_DIR
from classes.ner_markup_parser import NERMarkupParser


def analyze_ner_annotations():
    knowledge_path = os.path.join(BASE_DIR, 'knowledge', 'intents')
    parser = NERMarkupParser()

    total_examples = 0
    total_entities = 0
    entity_type_counts = Counter()
    intent_entity_map = {}

    print("=" * 80)
    print("ANALISI ANNOTAZIONI NER")
    print("=" * 80)
    print(f"\n📁 Percorso: {knowledge_path}")

    files = [f for f in os.listdir(knowledge_path) if f.endswith('.json')]
    print(f"📄 File trovati: {files}\n")

    for filename in sorted(files):
        filepath = os.path.join(knowledge_path, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠ Errore parsing {filename}: {e}, saltato")
            continue

        if 'nlu' not in data or 'intents' not in data['nlu']:
            print(f"⚠ {filename}: struttura non valida, saltato")
            continue

        print(f"\n📄 File: {filename}")

        for intent_data in data['nlu']['intents']:
            intent_name = intent_data['intent']
            examples = intent_data['examples']

            intent_entities = []
            intent_example_count = len(examples)
            total_examples += intent_example_count

            for example in examples:
                clean_text, entities = parser.parse(example)
                if entities:
                    total_entities += len(entities)
                    for ent in entities:
                        entity_type_counts[ent['entity']] += 1
                        intent_entities.append(ent['entity'])

            unique_entities = set(intent_entities)
            if unique_entities:
                intent_entity_map[intent_name] = {
                    'entities': list(unique_entities),
                    'count': len(intent_entities),
                    'examples': intent_example_count
                }
                print(f"   • {intent_name:25s}: {intent_example_count:3d} esempi, "
                      f"{len(intent_entities):3d} entità → {sorted(unique_entities)}")

    # Riepilogo globale
    print(f"\n{'='*80}")
    print("STATISTICHE GLOBALI")
    print(f"{'='*80}")
    print(f"\n📊 Totali:")
    print(f"   • Esempi totali: {total_examples}")
    print(f"   • Entità totali: {total_entities}")
    print(f"   • Intents con entità: {len(intent_entity_map)}")

    print(f"\n🏷️  Distribuzione tipi di entità:")
    for entity_type, count in entity_type_counts.most_common():
        percentage = (count / total_entities * 100) if total_entities > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"   {entity_type:12s}: {count:4d} ({percentage:5.1f}%) {bar}")

    # Intent con più entità
    print(f"\n🔝 Top 5 Intent per numero di entità:")
    sorted_intents = sorted(intent_entity_map.items(), key=lambda x: x[1]['count'], reverse=True)
    for intent_name, data in sorted_intents[:5]:
        print(f"   {intent_name:25s}: {data['count']:3d} entità in {data['examples']:3d} esempi")

    # Coverage
    annotated_examples = sum(1 for intent_name, data in intent_entity_map.items())
    print(f"\n📈 Coverage:")
    print(f"   • Esempi con entità: {sum(data['count'] for data in intent_entity_map.values())}")
    if total_examples > 0:
        print(f"   • Percentuale di annotazione: {(total_entities/total_examples)*100:.1f}%")
    else:
        print(f"   • Percentuale di annotazione: N/A (nessun esempio trovato)")

    return {
        'total_examples': total_examples,
        'total_entities': total_entities,
        'entity_types': dict(entity_type_counts),
        'intent_entity_map': intent_entity_map
    }


if __name__ == "__main__":
    stats = analyze_ner_annotations()

    print(f"\n{'='*80}")
    print("✅ Analisi completata!")
    print(f"{'='*80}")





