import json
import os
import csv
import numpy as np
import fasttext
from classes.intent_normalizer import IntentNormalizer
from classes.simple_tokenizer import SimpleTokenizer
from classes.ner_tag_builder import NERTagBuilder
from classes.ner_markup_parser import NERMarkupParser
from config import BASE_DIR


class ValidationDatasetGenerator:
    def __init__(self, validation_data_path):
        self.validation_data_path = validation_data_path
        self.data_path = os.path.join(BASE_DIR, '.cognitor')
        self.validation_data_output = os.path.join(BASE_DIR, 'training_data', 'validation')
        
        os.makedirs(self.validation_data_output, exist_ok=True)
        
        with open(self.validation_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.normalizer = IntentNormalizer()
        fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
        self.tokenizer = SimpleTokenizer(fasttext_model_path)
        self.ner_builder = NERTagBuilder.load(os.path.join(self.data_path, 'ner_tag_dict.json'))
        self.markup_parser = NERMarkupParser()
        
        with open(os.path.join(self.data_path, 'intent_dict.json'), 'r') as f:
            self.intent_dict = json.load(f)
        
        self.intent_to_id = {v: k for k, v in self.intent_dict.items()}

    def generate_validation_nlu(self):
        nlu_data = self.data['nlu']
        intents_data = nlu_data['intents']
        
        csv_path = os.path.join(self.validation_data_output, 'validation_nlu_data.csv')
        self._write_csv(csv_path, intents_data)
        self.tokenize_and_save_npy(csv_path)
        
    def _write_csv(self, csv_path, intents_data):
        seen = set()

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['INPUT', 'OUTPUT'])

            for intent_info in intents_data:
                intent_name = intent_info['intent']
                
                if intent_name not in self.intent_to_id:
                    print(f"Attenzione: intent '{intent_name}' non trovato nel dizionario, skippo...")
                    continue
                
                intent_id = self.intent_to_id[intent_name]

                for raw_example in intent_info['examples']:
                    clean_text, _ = self.markup_parser.parse(raw_example)
                    normalized = self.normalizer.normalize(clean_text)

                    for text in [clean_text, normalized]:
                        if text and text not in seen:
                            seen.add(text)
                            writer.writerow([text, intent_id])

    def tokenize_and_save_npy(self, csv_path):
        entity_map = self._build_entity_map()
        tokenized_data = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_text = row['INPUT']
                output_id = int(row['OUTPUT'])

                tokens = self.tokenizer(input_text)
                token_ids = [self.tokenizer.get_word_index(t) for t in tokens]

                entities = entity_map.get(input_text, [])
                bio_tag_ids = self.ner_builder.align_tokens_to_bio(input_text, tokens, entities)

                tokenized_data.append([token_ids, [output_id], bio_tag_ids])

        output_path = os.path.join(self.validation_data_output, 'tokenized_validation_data.npy')
        np.save(
            output_path,
            np.array(tokenized_data, dtype=object)
        )
        print(f"Dati di validazione tokenizzati salvati in: {output_path}")
        print(f"Numero di esempi di validazione: {len(tokenized_data)}")

    def _build_entity_map(self) -> dict:
        entity_map = {}
        for intent_info in self.data['nlu']['intents']:
            for raw_example in intent_info['examples']:
                clean_text, entities = self.markup_parser.parse(raw_example)
                if entities:
                    entity_map[clean_text] = entities
        return entity_map


if __name__ == "__main__":
    validation_data_path = os.path.join(BASE_DIR, 'training_data', 'validation', 'validation-intents.json')
    generator = ValidationDatasetGenerator(validation_data_path)
    generator.generate_validation_nlu()
