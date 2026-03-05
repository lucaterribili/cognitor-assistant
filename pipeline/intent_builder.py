import os
from classes.dataset_generator import DatasetGenerator


def build_intents(intents_dir: str = None, output_dir: str = "data"):
    """
    Genera i dataset NLU da file YAML.

    Args:
        intents_dir: Directory contenente i file YAML degli intents (default: knowledge/intents)
        output_dir: Directory di output per i file generati
    """
    # Carica da YAML
    dataset_generator = DatasetGenerator.load_from_yaml_files(intents_dir)
    dataset_generator.generate_nlu()
    print(f"Intent builder completato. File generati in: {output_dir}")


if __name__ == "__main__":
    build_intents()
