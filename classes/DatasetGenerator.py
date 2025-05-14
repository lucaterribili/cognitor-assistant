import json
import os
import csv

from classes.intent_normalizer import IntentNormalizer
from config import BASE_DIR

class DatasetGenerator:

    def __init__(self, data):
        self.data = data
        self.data_path = os.path.join(BASE_DIR, 'data')
        self.normalizer = IntentNormalizer()

    def generate_nlu(self):
        nlu_data = self.data['nlu']
        intents_data = nlu_data['intents']

        # Creazione del dizionario degli intenti con chiavi numeriche
        intent_dict = {i: intent['intent'] for i, intent in enumerate(intents_data)}
        intent_dict_path = os.path.join(self.data_path, 'intent_dict.json')
        with open(intent_dict_path, mode='w', encoding='utf-8') as json_file:
            json.dump(intent_dict, json_file, ensure_ascii=False, indent=4)

        # Creazione del file CSV
        csv_path = os.path.join(self.data_path, 'nlu_data.csv')
        os.makedirs(self.data_path, exist_ok=True)

        with open(csv_path, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['INPUT', 'OUTPUT'])

            for intent_id, intent in intent_dict.items():
                for example in next(item['examples'] for item in intents_data if item['intent'] == intent):
                    writer.writerow([self.normalizer.normalize(example), intent_id])

        print(f"Dizionario degli intenti: {intent_dict}")
        print(f"File CSV generato in: {csv_path}")