import json
import os
import csv
import numpy as np
import sentencepiece as spm
from classes.intent_normalizer import IntentNormalizer
from config import BASE_DIR, TOKENIZER_PATH

class DatasetGenerator:

    def __init__(self, data):
        self.data = data
        self.data_path = os.path.join(BASE_DIR, 'data')
        self.normalizer = IntentNormalizer()
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_file=TOKENIZER_PATH)

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
                tokens = self.sp_model.EncodeAsPieces(text)
                f.write(" ".join(tokens) + "\n")

        print(f"FastText corpus salvato in: {fasttext_path}")

    def tokenize_and_save_npy(self, csv_path):
        """
        Legge il file CSV generato da generate_nlu, tokenizza i dati e li salva in un file .npy.
        """
        tokenized_data = []

        # Legge il file CSV
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                input_text = row['INPUT']
                output_id = int(row['OUTPUT'])

                # Tokenizza input e output
                tokenized_input = self.sp_model.Encode(input_text, out_type=int)
                tokenized_output = [output_id]  # L'output è un ID numerico

                tokenized_data.append([tokenized_input, tokenized_output])

        # Conversione in array NumPy
        np_tokenized_data = np.array(tokenized_data, dtype=object)

        # Salvataggio in un file .npy
        npy_path = os.path.join(self.data_path, 'tokenized_data.npy')
        np.save(npy_path, np_tokenized_data)

        print(f"Dati tokenizzati salvati in: {npy_path}")