import os
import sys
from intellective.train_fast_text import train_embedder
from intellective.train_intent_classifier import train_main_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.intent_builder import build_intents
from pipeline.merge_data import merge_intents


def run_full_pipeline(
    data_path: str = "data/training_source.json",
    merge_data: bool = True,
    train_fasttext: bool = True,
    train_classifier: bool = True
):
    """
    Esegue la pipeline completa di training:
    1. Merge dei dati da training_data/ (opzionale)
    2. Genera i dati NLU (intent builder)
    3. Addestra FastText (opzionale)
    4. Addestra Intent Classifier (opzionale)
    """
    print("=" * 50)
    print("AVVIO PIPELINE COMPLETA")
    print("=" * 50)

    data_dir = os.path.join(BASE_DIR, "data")
    for item in os.listdir(data_dir):
        if item == ".gitkeep":
            continue
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            import shutil
            shutil.rmtree(item_path)

    if merge_data:
        print("\n[1/4] Merge dei dati da training_data/...")
        merge_intents(output_file=data_path)
    else:
        print("\n[1/4] Merge dei dati - SKIPPED")

    if train_fasttext:
        print("\n[2/4] Training FastText...")
        train_embedder()
    else:
        print("\n[2/4] Training FastText - SKIPPED")

    print("\n[3/4] Generazione dati NLU...")
    build_intents(data_path=data_path)

    if train_classifier:
        print("\n[4/4] Training Intent Classifier...")
        train_main_model()

    else:
        print("\n[4/4] Training Intent Classifier - SKIPPED")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETATA")
    print("=" * 50)


__all__ = ['run_full_pipeline', 'build_intents', 'merge_intents']
