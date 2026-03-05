import os
import sys
from intellective.train_fast_text import train_embedder
from intellective.train_intent_classifier import train_main_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.intent_builder import build_intents
from pipeline.merge_data import merge_intents


def run_full_pipeline(
    train_classifier: bool = True
):
    """
    Esegue la pipeline completa di training:
    1. Genera corpus FastText (raw text, senza tokenizer)
    2. Allena FastText (OBBLIGATORIO - il tokenizer ne ha bisogno)
    3. Genera dataset NLU tokenizzato (usando FastText appena addestrato)
    4. Allena Intent Classifier (opzionale)
    """
    print("=" * 50)
    print("AVVIO PIPELINE COMPLETA (YAML-based)")
    print("=" * 50)

    data_dir = os.path.join(BASE_DIR, ".cognitor")
    os.makedirs(data_dir, exist_ok=True)
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            import shutil
            shutil.rmtree(item_path)

    # STEP 1: Carica intents da YAML e genera corpus FastText (RAW TEXT)
    print("\n[1/4] Caricamento intents da YAML e generazione corpus FastText...")
    from classes.dataset_generator import DatasetGenerator
    generator = DatasetGenerator.load_from_yaml_files()
    generator.generate_fasttext_corpus_only()

    # STEP 2: Allena FastText (OBBLIGATORIO)
    print("\n[2/4] Training FastText (obbligatorio)...")
    train_embedder()

    # STEP 3: Genera dataset NLU tokenizzato (usa FastText appena addestrato)
    print("\n[3/4] Generazione dataset NLU tokenizzato...")
    build_intents()

    # STEP 4: Allena Intent Classifier
    if train_classifier:
        print("\n[4/4] Training Intent Classifier...")
        train_main_model()

    else:
        print("\n[4/4] Training Intent Classifier - SKIPPED")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETATA")
    print("=" * 50)


__all__ = ['run_full_pipeline', 'build_intents', 'merge_intents']
