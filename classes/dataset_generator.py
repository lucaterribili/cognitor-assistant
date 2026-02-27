import json
import os
import csv
import numpy as np
import fasttext
from classes.intent_normalizer import IntentNormalizer
from classes.simple_tokenizer import SimpleTokenizer
from config import BASE_DIR


class DatasetGenerator:

    def __init__(self, data):
        self.data = data
        self.data_path = os.path.join(BASE_DIR, 'data')
        self.normalizer = IntentNormalizer()
        fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
        self.tokenizer = SimpleTokenizer(fasttext_model_path)

    def generate_nlu(self):
        nlu_data = self.data['nlu']
        intents_data = nlu_data['intents']

        intent_dict = {i: intent['intent'] for i, intent in enumerate(intents_data)}
        intent_dict_path = os.path.join(self.data_path, 'intent_dict.json')
        with open(intent_dict_path, mode='w', encoding='utf-8') as json_file:
            json.dump(intent_dict, json_file, ensure_ascii=False, indent=4)

        csv_path = os.path.join(self.data_path, 'nlu_data.csv')
        os.makedirs(self.data_path, exist_ok=True)

        with open(csv_path, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['INPUT', 'OUTPUT'])

            seen = set()
            for intent_id, intent in intent_dict.items():
                examples = next(item['examples'] for item in intents_data if item['intent'] == intent)
                for example in examples:
                    normalized = self.normalizer.normalize(example)
                    for text in [example, normalized]:
                        if text and text not in seen:
                            seen.add(text)
                            writer.writerow([text, intent_id])

        self.tokenize_and_save_npy(csv_path)
        self.generate_fasttext_corpus()

    def generate_fasttext_corpus(self):
        fasttext_path = os.path.join(self.data_path, 'fast-text.txt')
        training_phrases_path = os.path.join(BASE_DIR, 'training_data', 'fasttext_phrases.txt')
        nlu_data = self.data['nlu']
        intents_data = nlu_data['intents']

        seen = set()
        lines_to_write = []

        for intent in intents_data:
            for example in intent['examples']:
                normalized = self.normalizer.normalize(example)
                for text in [example, normalized]:
                    if text and text not in seen:
                        seen.add(text)
                        lines_to_write.append(text)

        if os.path.exists(training_phrases_path):
            print(f"Merge con frasi da: {training_phrases_path}")
            with open(training_phrases_path, mode='r', encoding='utf-8') as phrases_file:
                for line in phrases_file:
                    line = line.strip()
                    if line and line not in seen:
                        seen.add(line)
                        lines_to_write.append(line)

        with open(fasttext_path, mode='w', encoding='utf-8') as f:
            for text in lines_to_write:
                tokens = self.tokenizer(text)
                f.write(" ".join(tokens) + "\n")

        print(f"FastText corpus salvato in: {fasttext_path}")

    def tokenize_and_save_npy(self, csv_path):
        tokenized_data = []

        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                input_text = row['INPUT']
                output_id = int(row['OUTPUT'])

                tokens = self.tokenizer(input_text)
                token_ids = [self.tokenizer.get_word_index(token) for token in tokens]
                tokenized_input = token_ids
                tokenized_output = [output_id]

                tokenized_data.append([tokenized_input, tokenized_output])

        np_tokenized_data = np.array(tokenized_data, dtype=object)

        npy_path = os.path.join(self.data_path, 'tokenized_data.npy')
        np.save(npy_path, np_tokenized_data)

        print(f"Dati tokenizzati salvati in: {npy_path}")
