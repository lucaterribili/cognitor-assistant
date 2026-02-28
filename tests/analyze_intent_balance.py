"""
Analisi del bilanciamento degli intenti nel dataset
"""

import json
import os
from collections import Counter
from config import BASE_DIR


def analyze_intent_balance():
    """
    Analizza la distribuzione degli intenti nel dataset per verificare il bilanciamento
    """
    knowledge_path = os.path.join(BASE_DIR, 'knowledge', 'intents')

    intent_counts = Counter()
    intent_examples = {}
    total_examples = 0

    print("=" * 80)
    print("ANALISI BILANCIAMENTO INTENTI")
    print("=" * 80)
    print(f"\n📁 Percorso: {knowledge_path}\n")

    # Leggi tutti i file JSON
    files = [f for f in os.listdir(knowledge_path) if f.endswith('.json')]
    print(f"📄 File trovati: {len(files)}\n")

    for filename in sorted(files):
        filepath = os.path.join(knowledge_path, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Errore parsing {filename}: {e}, saltato")
            continue

        if 'nlu' not in data or 'intents' not in data['nlu']:
            print(f"⚠️  {filename}: struttura non valida, saltato")
            continue

        print(f"📄 {filename}:")

        for intent_data in data['nlu']['intents']:
            intent_name = intent_data['intent']
            examples = intent_data['examples']
            num_examples = len(examples)

            intent_counts[intent_name] = num_examples
            intent_examples[intent_name] = examples
            total_examples += num_examples

            print(f"   • {intent_name:30s}: {num_examples:4d} esempi")

    # Statistiche globali
    print(f"\n{'=' * 80}")
    print("STATISTICHE GLOBALI")
    print(f"{'=' * 80}\n")

    num_intents = len(intent_counts)
    avg_examples = total_examples / num_intents if num_intents > 0 else 0

    print(f"📊 Totali:")
    print(f"   • Numero di intenti: {num_intents}")
    print(f"   • Esempi totali: {total_examples}")
    print(f"   • Media esempi per intent: {avg_examples:.1f}")

    # Distribuzione
    print(f"\n📈 Distribuzione (ordinata per frequenza):")
    print(f"{'Intent':<35s} {'Esempi':>8s} {'%':>7s} {'Deviazione':>12s} {'Grafico'}")
    print("-" * 80)

    for intent_name, count in intent_counts.most_common():
        percentage = (count / total_examples * 100) if total_examples > 0 else 0
        deviation = count - avg_examples
        deviation_pct = (deviation / avg_examples * 100) if avg_examples > 0 else 0

        # Grafico a barre
        bar_length = int(percentage * 2)  # Scala per visualizzazione
        bar = "█" * min(bar_length, 50)

        # Colore per deviazione
        if abs(deviation_pct) < 20:
            status = "✓"  # Bilanciato
        elif deviation_pct > 0:
            status = "↑"  # Sovra-rappresentato
        else:
            status = "↓"  # Sotto-rappresentato

        print(f"{intent_name:<35s} {count:>8d} {percentage:>6.1f}% {deviation:>+7.0f} ({deviation_pct:>+5.1f}%) {status} {bar}")

    # Analisi bilanciamento
    print(f"\n{'=' * 80}")
    print("ANALISI BILANCIAMENTO")
    print(f"{'=' * 80}\n")

    min_count = min(intent_counts.values()) if intent_counts else 0
    max_count = max(intent_counts.values()) if intent_counts else 0
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"📊 Metriche:")
    print(f"   • Intent con più esempi: {intent_counts.most_common(1)[0][0] if intent_counts else 'N/A'} ({max_count} esempi)")
    print(f"   • Intent con meno esempi: {intent_counts.most_common()[-1][0] if intent_counts else 'N/A'} ({min_count} esempi)")
    print(f"   • Rapporto di sbilanciamento: {imbalance_ratio:.2f}x")

    # Categorizzazione
    well_balanced = []
    over_represented = []
    under_represented = []

    for intent_name, count in intent_counts.items():
        deviation_pct = ((count - avg_examples) / avg_examples * 100) if avg_examples > 0 else 0

        if abs(deviation_pct) < 20:
            well_balanced.append(intent_name)
        elif deviation_pct > 0:
            over_represented.append(intent_name)
        else:
            under_represented.append(intent_name)

    print(f"\n🎯 Categorizzazione (soglia ±20%):")
    print(f"   • Bilanciati: {len(well_balanced)} intents")
    print(f"   • Sovra-rappresentati: {len(over_represented)} intents")
    print(f"   • Sotto-rappresentati: {len(under_represented)} intents")

    if under_represented:
        print(f"\n⚠️  Intents sotto-rappresentati (< {avg_examples * 0.8:.0f} esempi):")
        for intent_name in sorted(under_represented):
            count = intent_counts[intent_name]
            print(f"   • {intent_name:30s}: {count:4d} esempi (mancano ~{int(avg_examples - count)} esempi)")

    if over_represented:
        print(f"\n⚡ Intents sovra-rappresentati (> {avg_examples * 1.2:.0f} esempi):")
        for intent_name in sorted(over_represented):
            count = intent_counts[intent_name]
            print(f"   • {intent_name:30s}: {count:4d} esempi (eccedono ~{int(count - avg_examples)} esempi)")

    # Raccomandazioni
    print(f"\n{'=' * 80}")
    print("💡 RACCOMANDAZIONI")
    print(f"{'=' * 80}\n")

    if imbalance_ratio < 2:
        print("✅ Il dataset è ben bilanciato (rapporto < 2x)")
    elif imbalance_ratio < 5:
        print("⚠️  Il dataset ha uno sbilanciamento moderato (2x-5x)")
        print("   Considera di aggiungere più esempi agli intents sotto-rappresentati")
    else:
        print("❌ Il dataset è fortemente sbilanciato (> 5x)")
        print("   AZIONE RICHIESTA:")
        print("   1. Aggiungi esempi agli intents sotto-rappresentati")
        print("   2. Considera tecniche di data augmentation")
        print("   3. Usa class_weight nel training per compensare")

    if len(under_represented) > len(intent_counts) * 0.3:
        print(f"\n⚠️  Oltre il 30% degli intents è sotto-rappresentato")
        print("   Priorità: espandere questi intents per migliorare le performance")

    return {
        'intent_counts': dict(intent_counts),
        'total_examples': total_examples,
        'num_intents': num_intents,
        'avg_examples': avg_examples,
        'imbalance_ratio': imbalance_ratio,
        'well_balanced': well_balanced,
        'over_represented': over_represented,
        'under_represented': under_represented
    }


if __name__ == "__main__":
    stats = analyze_intent_balance()

    print(f"\n{'=' * 80}")
    print("✅ Analisi completata!")
    print(f"{'=' * 80}")