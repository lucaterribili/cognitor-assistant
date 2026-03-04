import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import fasttext

from config import BASE_DIR, DOPING_ACTIVE, MIN_INTENT_CONFIDENCE
from intellective.intent_classifier import IntentClassifier
from intellective.doping_preprocessor import DopingPreprocessor
from agent.session_manager import SessionManager
from agent.answer_manager import AnswerManager, SlotValidator


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = BASE_DIR
        self.session_manager = SessionManager()
        self.current_session_id = None
        
        self.fasttext_model_path = os.path.join(self.base_dir, 'models', 'fasttext_model.bin')
        self.intent_dict_path = os.path.join(self.base_dir, 'data', 'intent_dict.json')
        self.model_path = os.path.join(self.base_dir, 'models', 'intent_model_fast.pth')
        
        self.rules_base_path = os.path.join(self.base_dir, 'knowledge', 'rules')
        self.responses_base_path = os.path.join(self.base_dir, 'knowledge', 'responses')
        
        self.model = None
        self.intent_dict = None
        self.rules = {}
        self.responses = {}
        self.answer_manager = None
        self.doping_preprocessor = DopingPreprocessor()
        
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
            print(f"OK Modello trainato caricato")
            model_loaded = True
        except FileNotFoundError:
            print(f"WARNING Modello non trovato, usando pesi random")
        except Exception as e:
            print(f"WARNING Errore caricamento modello: {e}")
            print(f"WARNING Usando pesi random")
        
        self.model.to(self.device)
        self.model.eval()
        
        return model_loaded
    
    def load_knowledge(self):
        print("Caricamento rules e responses...")
        
        rules_dir = os.path.join(self.base_dir, 'knowledge', 'rules')
        responses_dir = os.path.join(self.base_dir, 'knowledge', 'responses')
        
        for filename in os.listdir(rules_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(rules_dir, filename)
                with open(file_path, 'r') as f:
                    rules_data = json.load(f)
                    self.rules.update(rules_data.get('rules', {}))
        
        for filename in os.listdir(responses_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(responses_dir, filename)
                with open(file_path, 'r') as f:
                    responses_data = json.load(f)
                    self.responses.update(responses_data.get('responses', {}))
        
        self.answer_manager = AnswerManager(self.rules)
        self.slot_validator = SlotValidator(self.rules)
        
        print(f"OK Caricate {len(self.rules)} rules e {len(self.responses)} response keys")
        
        print("Costruzione lookup table per doping...")
        nlu_dir = os.path.join(self.base_dir, 'knowledge', 'intents')
        for filename in os.listdir(nlu_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(nlu_dir, filename)
                with open(file_path, 'r') as f:
                    nlu_data = json.load(f)
                    self.doping_preprocessor.build_lookup_table(nlu_data)
        
        print("OK Lookup table costruita")
    
    def get_response(self, intent_name, slots: dict = None):
        if slots is None:
            slots = {}
        
        return self.answer_manager.get_response(intent_name, slots, self.responses)
    
    def predict(self, text):
        if DOPING_ACTIVE:
            doped_text = self.doping_preprocessor.dope_input(text)
            text_to_predict = doped_text
            is_doped = doped_text != text
        else:
            text_to_predict = text
            is_doped = False
        
        result = self.model.predict(text_to_predict)
        intent_idx = result['intent_idx']
        intent_name = self.intent_dict[str(intent_idx)]
        confidence = result['intent_confidence']
        
        # Se la confidenza è troppo bassa, usa un fallback generico
        if confidence < MIN_INTENT_CONFIDENCE:
            intent_name = 'low_confidence_fallback'

        return {
            'intent': intent_name,
            'confidence': confidence,
            'entities': result.get('entities', []),
            'doped': is_doped
        }
    
    def chat(self):
        self.current_session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(self.current_session_id)
        
        print("\n" + "="*50)
        print("ARIANNA AGENT - Interfaccia Testuale")
        print("="*50)
        print(f"Session ID: {self.current_session_id}")
        print(f"Sessioni attive: {len(self.session_manager.get_active_sessions())}")
        print("Scrivi un messaggio (o 'esci' per terminare)\n")
        
        while True:
            mode_indicator = f"[{session.agent_mode.upper()}] " if session.agent_mode != "predictable" else ""
            try:
                user_input = input(f"Tu: {mode_indicator}").strip()
            except EOFError:
                break
            
            if user_input.lower() in ['esci', 'exit', 'quit', 'q']:
                print("\nArrivederci!")
                break
            
            if not user_input:
                continue
            
            if session.agent_mode == "inputable":
                if session.waiting_for_slot:
                    slot_name = session.waiting_for_slot["slot"]
                    pending_intent = session.waiting_for_slot["intent"]
                    
                    exit_commands = ['#exit', '#annulla', '#cancel', '#abort']
                    if user_input.lower() in exit_commands:
                        print("\nArianna: Input annullato. Puoi fornire un nuovo comando.\n")
                        session.waiting_for_slot = None
                        session.agent_mode = "predictable"
                        session.add_message("user", user_input)
                        session.add_message("assistant", "Input annullato.", None)
                        continue

                    if not self.slot_validator.validate(pending_intent, slot_name, user_input):
                        print("\nArianna: Selezione non valida. Riprova.\n")
                        session.add_message("user", user_input)
                        continue

                    session.update_context(slot_name, user_input)
                    session.waiting_for_slot = None
                    session.agent_mode = "predictable"

                    response, wait_for_slot = self.get_response(pending_intent, session.context)
                    print(f"\nArianna: {response}\n")

                    if wait_for_slot:
                        session.waiting_for_slot = {"intent": pending_intent, "slot": wait_for_slot}

                    session.add_message("user", user_input)
                    session.add_message("assistant", response, pending_intent)
                    continue
            
            prediction = self.predict(user_input)
            
            print(f"\nIntent: {prediction['intent']} ({prediction['confidence']:.1%})")
            
            entities_str = ', '.join([e['value'] for e in prediction['entities']]) or "nessuna"
            print(f"Entita: {entities_str}")
            
            for entity in prediction.get('entities', []):
                session.update_context(entity['entity'], entity['value'])
            
            response, wait_for_slot = self.get_response(prediction['intent'], session.context)
            print(f"\nArianna: {response}\n")
            
            if wait_for_slot:
                session.waiting_for_slot = {"intent": prediction['intent'], "slot": wait_for_slot}
                session.agent_mode = "inputable"
            
            session.add_message("user", user_input, prediction['intent'], prediction.get('entities', []))
            session.add_message("assistant", response, prediction['intent'])
            
            print(f"Cronologia: {len(session.history)} messaggi | Contesto: {session.context}")


def main():
    agent = Agent()
    
    agent.load_models()
    agent.load_knowledge()
    
    agent.chat()


if __name__ == "__main__":
    main()
