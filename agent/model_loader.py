"""
Gestisce il caricamento dei modelli ML e knowledge base.
"""
import json
import os
from typing import Dict

import fasttext
import torch
import yaml

from intellective.intent_classifier import IntentClassifier
from intellective.doping_preprocessor import DopingPreprocessor


class ModelLoader:
    """Carica e gestisce i modelli ML."""

    def __init__(self, base_dir: str, device: torch.device):
        self.base_dir = base_dir
        self.device = device

        self.fasttext_model_path = os.path.join(base_dir, 'models', 'fasttext_model.bin')
        self.intent_dict_path = os.path.join(base_dir, 'data', 'intent_dict.json')
        self.model_path = os.path.join(base_dir, 'models', 'intent_model_fast.pth')

    def load_fasttext(self) -> fasttext.FastText._FastText:
        """Carica il modello FastText."""
        print("Caricamento modello FastText...")
        return fasttext.load_model(self.fasttext_model_path)

    def load_intent_dict(self) -> Dict[str, str]:
        """Carica il dizionario degli intent."""
        print("Caricamento intent dictionary...")
        with open(self.intent_dict_path, 'r') as f:
            return json.load(f)

    def load_intent_classifier(self, ft_model: fasttext.FastText._FastText,
                               intents_number: int) -> tuple[IntentClassifier, bool]:
        """
        Carica il modello Intent Classifier.

        Returns:
            tuple: (modello, success_flag)
        """
        print("Caricamento modello Intent Classifier...")
        vocab_size = len(ft_model.words)

        model = IntentClassifier(
            vocab_size=vocab_size,
            embed_dim=300,
            hidden_dim=256,
            output_dim=intents_number,
            dropout_prob=0.3,
            fasttext_model_path=self.fasttext_model_path,
            freeze_embeddings=True
        )

        model_loaded = False
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("OK Modello trainato caricato")
            model_loaded = True
        except FileNotFoundError:
            print("WARNING Modello non trovato, usando pesi random")
        except Exception as e:
            print(f"WARNING Errore caricamento modello: {e}")
            print("WARNING Usando pesi random")

        model.to(self.device)
        model.eval()

        return model, model_loaded


class KnowledgeLoader:
    """Carica rules, responses e costruisce lookup table per doping."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.rules_dir = os.path.join(base_dir, 'knowledge', 'rules')
        self.responses_dir = os.path.join(base_dir, 'knowledge', 'responses')
        self.intents_dir = os.path.join(base_dir, 'knowledge', 'intents')

    def load_rules(self) -> Dict[str, dict]:
        """
        Carica tutte le rules dai file JSON o YAML.
        Priorità: YAML > JSON (se esiste YAML, usa quello)
        """
        rules = {}
        for filename in os.listdir(self.rules_dir):
            # Priorità a YAML
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                file_path = os.path.join(self.rules_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    rules_data = yaml.safe_load(f)
                    rules.update(rules_data.get('rules', {}))
            # Supporto legacy JSON (se non c'è YAML)
            elif filename.endswith('.json'):
                yaml_name = filename.replace('.json', '.yaml')
                yaml_path = os.path.join(self.rules_dir, yaml_name)
                # Salta JSON se esiste già il corrispondente YAML
                if not os.path.exists(yaml_path):
                    file_path = os.path.join(self.rules_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        rules_data = json.load(f)
                        rules.update(rules_data.get('rules', {}))
        return rules

    def load_responses(self) -> Dict[str, list]:
        """
        Carica tutte le responses dai file JSON o YAML.
        Priorità: YAML > JSON (se esiste YAML, usa quello)
        """
        responses = {}
        for filename in os.listdir(self.responses_dir):
            # Priorità a YAML
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                file_path = os.path.join(self.responses_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    responses_data = yaml.safe_load(f)
                    responses.update(responses_data.get('responses', {}))
            # Supporto legacy JSON (se non c'è YAML)
            elif filename.endswith('.json'):
                yaml_name = filename.replace('.json', '.yaml')
                yaml_path = os.path.join(self.responses_dir, yaml_name)
                # Salta JSON se esiste già il corrispondente YAML
                if not os.path.exists(yaml_path):
                    file_path = os.path.join(self.responses_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        responses_data = json.load(f)
                        responses.update(responses_data.get('responses', {}))
        return responses

    def build_doping_lookup_table(self, doping_preprocessor: DopingPreprocessor) -> None:
        """Costruisce la lookup table per il doping preprocessor."""
        print("Costruzione lookup table per doping...")
        for filename in os.listdir(self.intents_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.intents_dir, filename)
                with open(file_path, 'r') as f:
                    nlu_data = json.load(f)
                    doping_preprocessor.build_lookup_table(nlu_data)
        print("OK Lookup table costruita")

    def load_all(self) -> tuple[Dict[str, dict], Dict[str, list]]:
        """
        Carica rules e responses.

        Returns:
            tuple: (rules, responses)
        """
        print("Caricamento rules e responses...")
        rules = self.load_rules()
        responses = self.load_responses()
        print(f"OK Caricate {len(rules)} rules e {len(responses)} response keys")
        return rules, responses

