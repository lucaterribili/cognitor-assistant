import os
import sys
from intellective.train_fast_text import train_embedder
from intellective.train_intent_classifier import train_main_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from pipeline.intent_builder import build_intents
from pipeline.merge_data import merge_intents
from pipeline.validator import DatasetValidator


def run_full_pipeline(
    train_classifier: bool = True
):
    """
    Esegue la pipeline completa di training:
    0. Valida il dataset (intenti ed entità NER)
    1. Genera corpus FastText (raw text, senza tokenizer)
    2. Allena FastText (OBBLIGATORIO - il tokenizer ne ha bisogno)
    3. Genera dataset NLU tokenizzato (usando FastText appena addestrato)
    4. Allena Intent Classifier (opzionale)
    """
    print("=" * 50)
    print("AVVIO PIPELINE COMPLETA (YAML-based)")
    print("=" * 50)

    # STEP 0: Valida il dataset
    print("\n[0/5] Validazione dataset...")
    knowledge_path = os.path.join(BASE_DIR, 'knowledge')
    training_path = os.path.join(BASE_DIR, 'training_data')

    validator = DatasetValidator(
        knowledge_path=knowledge_path,
        training_path=training_path
    )

    is_valid = validator.validate_all()

    if not is_valid:
        print("\n⛔ VALIDAZIONE FALLITA - Impossibile procedere con il training")
        print("💡 Correggi gli errori riportati sopra e riprova.")
        return False

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
    print("\n[1/5] Caricamento intents da YAML e generazione corpus FastText...")
    from classes.dataset_generator import DatasetGenerator
    generator = DatasetGenerator.load_from_yaml_files()
    generator.generate_fasttext_corpus_only()

    # STEP 2: Allena FastText (OBBLIGATORIO)
    print("\n[2/5] Training FastText (obbligatorio)...")
    train_embedder()

    # STEP 3: Genera dataset NLU tokenizzato (usa FastText appena addestrato)
    print("\n[3/5] Generazione dataset NLU tokenizzato...")
    build_intents()

    # STEP 4: Allena Intent Classifier
    if train_classifier:
        print("\n[4/5] Training Intent Classifier...")
        train_main_model()

    else:
        print("\n[4/5] Training Intent Classifier - SKIPPED")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETATA")
    print("=" * 50)

    return True


__all__ = ['run_full_pipeline', 'build_intents', 'merge_intents']
