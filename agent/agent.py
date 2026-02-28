import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import fasttext

from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = BASE_DIR
        
        self.fasttext_model_path = os.path.join(self.base_dir, 'models', 'fasttext_model.bin')
        self.intent_dict_path = os.path.join(self.base_dir, 'data', 'intent_dict.json')
        self.model_path = os.path.join(self.base_dir, 'models', 'intent_model_fast.pth')
        
        self.rules_base_path = os.path.join(self.base_dir, 'knowledge', 'rules', 'base.json')
        self.rules_domain_path = os.path.join(self.base_dir, 'knowledge', 'rules', 'domain.json')
        self.responses_base_path = os.path.join(self.base_dir, 'knowledge', 'responses', 'base.json')
        self.responses_domain_path = os.path.join(self.base_dir, 'knowledge', 'responses', 'domain.json')
        
        self.model = None
        self.intent_dict = None
        self.rules = {}
        self.responses = {}
        
    def load_models(self):
        print("Caricamento modello FastText...")
        self.ft_model = fasttext.load_model(self.fasttext_model_path)
        vocab_size = len(self.ft_model.words)
        
        print("Caricamento intent dictionary...")
        with open(self.intent_dict_path, 'r') as f:
            self.intent_dict = json.load(f)
            intents_number = len(self.intent_dict)
        
        print("Caricamento modello Intent Classifier...")
        self.model = IntentClassifier(
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
            self.model.load_state_dict(state_dict)
            print(f"✓ Modello trainato caricato")
            model_loaded = True
        except FileNotFoundError:
            print(f"⚠️ Modello non trovato, usando pesi random")
        except Exception as e:
            print(f"⚠️ Errore caricamento modello: {e}")
            print(f"⚠️ Usando pesi random")
        
        self.model.to(self.device)
        self.model.eval()
        
        return model_loaded
    
    def load_knowledge(self):
        print("Caricamento rules e responses...")
        
        with open(self.rules_base_path, 'r') as f:
            rules_base = json.load(f)
            self.rules.update(rules_base.get('rules', {}))
        
        with open(self.rules_domain_path, 'r') as f:
            rules_domain = json.load(f)
            self.rules.update(rules_domain.get('rules', {}))
        
        with open(self.responses_base_path, 'r') as f:
            responses_base = json.load(f)
            self.responses.update(responses_base.get('responses', {}))
        
        with open(self.responses_domain_path, 'r') as f:
            responses_domain = json.load(f)
            self.responses.update(responses_domain.get('responses', {}))
        
        print(f"✓ Caricate {len(self.rules)} rules e {len(self.responses)} response keys")
    
    def get_response(self, intent_name):
        response_keys = self.rules.get(intent_name, [])
        
        if not response_keys:
            return "Non ho una risposta per questo. Puoi riformulare?"
        
        response_key = response_keys[0]
        response_list = self.responses.get(response_key, [])
        
        if not response_list:
            return f"Risposta non definita per {response_key}"
        
        return random.choice(response_list)
    
    def predict(self, text):
        result = self.model.predict(text)
        intent_idx = result['intent_idx']
        intent_name = self.intent_dict[str(intent_idx)]
        confidence = result['intent_confidence']
        
        return {
            'intent': intent_name,
            'confidence': confidence,
            'entities': result.get('entities', [])
        }
    
    def chat(self):
        print("\n" + "="*50)
        print("🤖 ARIANNA AGENT - Interfaccia Testuale")
        print("="*50)
        print("Scrivi un messaggio (o 'esci' per terminare)\n")
        
        while True:
            try:
                user_input = input("Tu: ").strip()
            except EOFError:
                break
            
            if user_input.lower() in ['esci', 'exit', 'quit', 'q']:
                print("\n👋 Arrivederci!")
                break
            
            if not user_input:
                continue
            
            prediction = self.predict(user_input)
            
            print(f"\n🎯 Intent: {prediction['intent']} ({prediction['confidence']:.1%})")
            
            if prediction['entities']:
                print(f"🏷️  Entità: {', '.join([e['value'] for e in prediction['entities']])}")
            
            response = self.get_response(prediction['intent'])
            print(f"\n🤖 Arianna: {response}\n")


def main():
    agent = Agent()
    
    agent.load_models()
    agent.load_knowledge()
    
    agent.chat()


if __name__ == "__main__":
    main()
