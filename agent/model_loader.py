"""
Gestisce il caricamento dei modelli ML e knowledge base.
"""
import json
import os
from typing import Dict

import torch
import yaml

from config import DOPING_ACTIVE, USE_FASTTEXT
from classes.intent_factorizer import IntentFactorizer
from intellective.intent_classifier import IntentClassifier
from intellective.doping_preprocessor import DopingPreprocessor


class ModelLoader:
    """Carica e gestisce i modelli ML."""

    def __init__(self, base_dir: str, device: torch.device):
        self.base_dir = base_dir
        self.device = device

        self.vocab_path = os.path.join(base_dir, '.cognitor', 'vocab.json')
        self.intent_dict_path = os.path.join(base_dir, '.cognitor', 'intent_dict.json')
        self.model_path = os.path.join(base_dir, 'models', 'intent_model_fast.pth')
        self.intent_factors_path = os.path.join(base_dir, '.cognitor', 'intent_factors.json')

    @staticmethod
    def _load_json(path: str, label: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            if not DOPING_ACTIVE:  # il doping è spento: la lookup table non serve e i dati dei clienti stanno in training_data/
                print(f"Lookup table doping non costruita: {self.intents_dir} non esiste (doping non attivo)")
                return
            raise FileNotFoundError(
                f"{label} non trovato in {path} — "
                "esegui prima la pipeline di training: python -m pipeline"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"{label} in {path} non è un JSON valido: {e}")

    def load_vocab_size(self) -> int:
        """Carica la dimensione del vocabolario da vocab.json."""
        print("Caricamento vocabolario...")
        return len(self._load_json(self.vocab_path, "vocab.json"))

    def load_intent_dict(self) -> Dict[str, str]:
        """Carica il dizionario degli intent."""
        print("Caricamento intent dictionary...")
        return self._load_json(self.intent_dict_path, "intent_dict.json")

    def load_intent_classifier(self, vocab_size: int, intents_number: int) -> tuple[IntentClassifier, bool]:
        """
        Carica il modello Intent Classifier.

        Returns:
            tuple: (modello, success_flag)
        """
        print("Caricamento modello Intent Classifier...")

        wordvectors_path = os.path.join(self.base_dir, '.cognitor', 'wordvectors.vec')
        sentencepiece_model_path = None if USE_FASTTEXT else os.path.join(
            self.base_dir, '.cognitor', 'sentencepiece.model'
        )

        # Il checkpoint decide la struttura: a due teste (argomento + tipo di entità) se contiene fc_topic.
        state_dict = None
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
        except FileNotFoundError:
            print("WARNING Modello non trovato, usando pesi random")
        except Exception as e:
            print(f"WARNING Errore caricamento modello: {e}")
            print("WARNING Usando pesi random")

        intent_factors = None
        if state_dict is not None and 'fc_topic.weight' in state_dict:
            intent_factors = IntentFactorizer.load(self.intent_factors_path)

        model = IntentClassifier(
            vocab_size=vocab_size,
            embed_dim=300,
            hidden_dim=256,
            output_dim=intents_number,
            dropout_prob=0.3,
            wordvectors_path=wordvectors_path,
            vocab_path=self.vocab_path,
            freeze_embeddings=True,
            sentencepiece_model_path=sentencepiece_model_path,
            intent_factors=intent_factors,
        )

        model_loaded = False
        if state_dict is not None:
            try:
                model.load_state_dict(state_dict)
                print("OK Modello trainato caricato")
                model_loaded = True
            except Exception as e:
                print(f"WARNING Errore caricamento modello: {e}")
                print("WARNING Usando pesi random")

        model.to(self.device)
        model.eval()

        return model, model_loaded


class KnowledgeLoader:
    """Carica rules, responses, conversations e costruisce la lookup table per doping."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.cognitor_dir = os.path.join(base_dir, '.cognitor')
        self.rules_file = os.path.join(self.cognitor_dir, 'rules.yaml')
        self.responses_file = os.path.join(self.cognitor_dir, 'responses.yaml')
        self.conversations_file = os.path.join(self.cognitor_dir, 'conversations.yaml')
        self.intents_dir = os.path.join(base_dir, 'knowledge', 'intents')

        self.legacy_rules_dir = os.path.join(base_dir, 'knowledge', 'rules')
        self.legacy_responses_dir = os.path.join(base_dir, 'knowledge', 'responses')
        self.legacy_conversations_dir = os.path.join(base_dir, 'knowledge', 'conversations')

    @staticmethod
    def _load_yaml_section(merged_file: str, legacy_dir: str, section_key: str) -> dict:
        """Carica una sezione YAML dal file mergiato in .cognitor/, con fallback
        alla cartella di file separati in knowledge/<section> (modalità legacy)."""
        if os.path.exists(merged_file):
            print(f"Caricamento {section_key} da: {merged_file}")
            with open(merged_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                return data.get(section_key, {})

        print(f"File {merged_file} non trovato, usando modalità legacy da {legacy_dir}")
        merged = {}
        if os.path.exists(legacy_dir):
            for filename in os.listdir(legacy_dir):
                if filename.endswith(('.yaml', '.yml')):
                    file_path = os.path.join(legacy_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f) or {}
                        merged.update(data.get(section_key, {}))
        return merged

    def load_rules(self) -> Dict[str, dict]:
        return self._load_yaml_section(self.rules_file, self.legacy_rules_dir, 'rules')

    def load_responses(self) -> Dict[str, list]:
        return self._load_yaml_section(self.responses_file, self.legacy_responses_dir, 'responses')

    def load_conversations(self) -> Dict[str, dict]:
        return self._load_yaml_section(self.conversations_file, self.legacy_conversations_dir, 'conversations')

    def build_doping_lookup_table(self, doping_preprocessor: DopingPreprocessor) -> None:
        """Costruisce la lookup table per il doping preprocessor."""
        print("Costruzione lookup table per doping...")
        try:
            filenames = os.listdir(self.intents_dir)
        except FileNotFoundError:
            if not DOPING_ACTIVE:  # doping spento: la lookup table non serve e i dati dei clienti stanno in training_data/
                print(f"Lookup table doping non costruita: {self.intents_dir} non esiste (doping non attivo)")
                return
            raise FileNotFoundError(
                f"Directory intents non trovata in {self.intents_dir} — "
                "esegui prima la pipeline di training: python -m pipeline"
            )
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(self.intents_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        nlu_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{file_path} non è un JSON valido: {e}")
                doping_preprocessor.build_lookup_table(nlu_data)
        print("OK Lookup table costruita")

    def load_all(self) -> tuple[Dict[str, dict], Dict[str, list], Dict[str, dict]]:
        """Carica rules, responses e conversations."""
        print("Caricamento rules, responses e conversations...")
        rules = self.load_rules()
        responses = self.load_responses()
        conversations = self.load_conversations()
        print(f"OK Caricate {len(rules)} rules, {len(responses)} response keys e {len(conversations)} conversations")
        return rules, responses, conversations
