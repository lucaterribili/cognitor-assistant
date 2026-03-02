import json
import os
import csv
import numpy as np
from classes.intent_normalizer import IntentNormalizer
from classes.simple_tokenizer import SimpleTokenizer
from classes.ner_markup_parser import NERMarkupParser
from classes.ner_tag_builder import NERTagBuilder
from intellective.doping_preprocessor import DopingPreprocessor
from config import BASE_DIR, DOPING_ACTIVE


class DatasetGenerator:

    def __init__(self, data):
        self.data = data
        self.data_path = os.path.join(BASE_DIR, 'data')
        self.normalizer = IntentNormalizer()
        fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
        self.tokenizer = SimpleTokenizer(fasttext_model_path)
        self.ner_parser = NERMarkupParser()
        self.ner_tag_builder = NERTagBuilder()
        self.doping_preprocessor = DopingPreprocessor()
        self.doping_preprocessor.build_lookup_table(data)

    def generate_nlu(self):
        nlu_data = self.data['nlu']
        intents_data = nlu_data['intents']

        intent_dict = {i: intent['intent'] for i, intent in enumerate(intents_data)}
        intent_dict_path = os.path.join(self.data_path, 'intent_dict.json')
        with open(intent_dict_path, mode='w', encoding='utf-8') as json_file:
            json.dump(intent_dict, json_file, ensure_ascii=False, indent=4)

        csv_path = os.path.join(self.data_path, 'nlu_data.csv')
        os.makedirs(self.data_path, exist_ok=True)
        
        if DOPING_ACTIVE:
            doped_dataset = self.doping_preprocessor.process_dataset(self.data)
        else:
            doped_dataset = self.doping_preprocessor.get_examples(self.data)

        with open(csv_path, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['INPUT', 'OUTPUT', 'CLEAN_TEXT', 'ENTITIES'])

            seen = set()
            intent_name_to_id = {v: k for k, v in intent_dict.items()}
            for item in doped_dataset:
                text = item['text']
                intent = item['intent']
                intent_id = intent_name_to_id.get(intent)
                if intent_id is None:
                    continue
                    
                clean_text, entities = self.ner_parser.parse(text)
                normalized = self.normalizer.normalize(clean_text)

                for text_variant in [clean_text, normalized]:
                    if text_variant and text_variant not in seen:
                        seen.add(text_variant)
                        writer.writerow([
                            text_variant,
                            intent_id,
                            clean_text,
                            json.dumps(entities, ensure_ascii=False)
                        ])

        self.tokenize_and_save_npy(csv_path)
        self.generate_fasttext_corpus()

        # Salva il tag builder
        tag_builder_path = os.path.join(self.data_path, 'ner_tag_builder.json')
        self.ner_tag_builder.save(tag_builder_path)
        print(f"NER tag builder salvato in: {tag_builder_path}")

    def generate_fasttext_corpus(self):
        fasttext_path = os.path.join(self.data_path, 'fast-text.txt')
        training_phrases_path = os.path.join(BASE_DIR, 'knowledge', 'embeddings.txt')
        
        if DOPING_ACTIVE:
            doped_dataset = self.doping_preprocessor.process_dataset(self.data)
        else:
            doped_dataset = self.doping_preprocessor.get_examples(self.data)

        seen = set()
        lines_to_write = []

        for item in doped_dataset:
            text = item['text']
            clean_text, _ = self.ner_parser.parse(text)
            normalized = self.normalizer.normalize(clean_text)
            for text_variant in [clean_text, normalized]:
                if text_variant and text_variant not in seen:
                    seen.add(text_variant)
                    lines_to_write.append(text_variant)

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
                clean_text = row['CLEAN_TEXT']
                entities = json.loads(row['ENTITIES'])

                # Tokenizza
                tokens = self.tokenizer(input_text)
                token_ids = [self.tokenizer.get_word_index(token) for token in tokens]

                # Genera tag NER BIO
                ner_tag_ids = self.ner_tag_builder.align_tokens_to_bio(clean_text, tokens, entities)

                # Salva: [token_ids, intent_id, ner_tag_ids]
                tokenized_data.append([token_ids, [output_id], ner_tag_ids])

        np_tokenized_data = np.array(tokenized_data, dtype=object)

        npy_path = os.path.join(self.data_path, 'tokenized_data.npy')
        np.save(npy_path, np_tokenized_data)

        print(f"Dati tokenizzati salvati in: {npy_path}")
