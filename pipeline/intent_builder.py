import json
import os
from classes.dataset_generator import DatasetGenerator


def build_intents(data_path: str = "data/training_source.json", output_dir: str = "data"):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    dataset_generator = DatasetGenerator(data)
    dataset_generator.generate_nlu()
    print(f"Intent builder completato. File generati in: {output_dir}")


if __name__ == "__main__":
    build_intents()
