#!/usr/bin/env python3
"""
Script per verificare e pulire i file FastText dopo l'estrazione dei vectors.

Verifica che:
1. vocab.json esiste ed è valido
2. wordvectors.vec esiste ed è valido
3. I vectors hanno le dimensioni corrette

Se tutto è OK, offre di eliminare il file .bin per risparmiare spazio.
"""

import os
import json
from config import BASE_DIR


def check_vocab():
    """Verifica che vocab.json esista ed è valido."""
    vocab_path = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')

    if not os.path.exists(vocab_path):
        return False, "vocab.json non trovato"

    try:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        if not isinstance(vocab, list):
            return False, "vocab.json non è una lista"

        if len(vocab) == 0:
            return False, "vocab.json è vuoto"

        return True, f"✓ vocab.json OK ({len(vocab)} parole)"

    except Exception as e:
        return False, f"Errore lettura vocab.json: {e}"


def check_wordvectors():
    """Verifica che wordvectors.vec esista ed è valido."""
    vec_path = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')

    if not os.path.exists(vec_path):
        return False, "wordvectors.vec non trovato"

    try:
        with open(vec_path, 'r') as f:
            # Prima riga: num_words embed_dim
            first_line = f.readline().strip().split()
            if len(first_line) != 2:
                return False, "Prima riga wordvectors.vec malformata"

            num_words = int(first_line[0])
            embed_dim = int(first_line[1])

            # Verifica qualche riga
            lines_checked = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) != embed_dim + 1:  # parola + vettore
                    return False, f"Riga {lines_checked+2} ha dimensione sbagliata"

                lines_checked += 1
                if lines_checked >= 10:  # Verifica solo le prime 10 righe
                    break

        return True, f"✓ wordvectors.vec OK ({num_words} parole, dim={embed_dim})"

    except Exception as e:
        return False, f"Errore lettura wordvectors.vec: {e}"


def get_file_size(path):
    """Ritorna la dimensione del file in formato human-readable."""
    size = os.path.getsize(path)

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0

    return f"{size:.1f} TB"


def main():
    print("=" * 80)
    print("VERIFICA E PULIZIA FASTTEXT")
    print("=" * 80)

    # Verifica vocab
    vocab_ok, vocab_msg = check_vocab()
    print(f"\n1. Vocabolario:")
    print(f"   {vocab_msg}")

    # Verifica vectors
    vectors_ok, vectors_msg = check_wordvectors()
    print(f"\n2. Word Vectors:")
    print(f"   {vectors_msg}")

    # Verifica file .bin
    bin_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
    bin_exists = os.path.exists(bin_path)
    bin_size = None

    print(f"\n3. FastText .bin:")
    if bin_exists:
        bin_size = get_file_size(bin_path)
        print(f"   ✓ Trovato: {bin_path}")
        print(f"   📦 Dimensione: {bin_size}")
    else:
        print(f"   ✗ Non trovato (già eliminato)")

    print("\n" + "=" * 80)

    # Decisione
    if not vocab_ok or not vectors_ok:
        print("❌ ERRORE: vocab.json o wordvectors.vec non validi!")
        print("   NON eliminare il file .bin")
        print("   Esegui di nuovo: python -m intellective.train_fast_text")
        return

    if not bin_exists:
        print("✓ Tutto OK! File .bin già eliminato.")
        return

    print("✓ Tutto OK! vocab.json e wordvectors.vec sono validi.")
    print(f"\n💾 Il file .bin occupa {bin_size} di spazio.")
    print("   Il sistema ora usa solo vocab.json e wordvectors.vec")
    print("   per l'inferenza (niente più FastText!).")

    response = input("\n🗑️  Vuoi eliminare fasttext_model.bin? [s/N]: ")

    if response.lower() in ['s', 'si', 'sì', 'y', 'yes']:
        try:
            os.remove(bin_path)
            print(f"\n✓ File eliminato: {bin_path}")
            print(f"✓ Spazio liberato: {bin_size}")
        except Exception as e:
            print(f"\n❌ Errore durante l'eliminazione: {e}")
    else:
        print("\n📌 File .bin mantenuto.")
        print("   Puoi eliminarlo manualmente quando vuoi:")
        print(f"   rm {bin_path}")


if __name__ == '__main__':
    main()


