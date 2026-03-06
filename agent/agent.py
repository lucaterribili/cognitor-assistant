"""
Agent principale per il chatbot Cognitor.
Coordina i modelli ML, la gestione delle sessioni e le risposte.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import BASE_DIR, DOPING_ACTIVE, MIN_INTENT_CONFIDENCE
from intellective.doping_preprocessor import DopingPreprocessor
from agent.session_manager import SessionManager
from agent.answer_manager import AnswerManager, SlotValidator
from agent.model_loader import ModelLoader, KnowledgeLoader
from agent.slot_manager import SlotManager
from agent.rule_interpreter import RuleInterpreter
from agent.operations.manager import OperationManager
from agent.conversation_balancer import ConversationBalancer


class Agent:
    """
    Agent principale che coordina tutti i componenti del chatbot.

    Responsabilità:
    - Caricamento e gestione dei modelli ML
    - Predizione di intent ed entità
    - Gestione delle risposte
    - Coordinamento con SessionManager
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = BASE_DIR

        # Manager e componenti
        self.session_manager = SessionManager()
        self.doping_preprocessor = DopingPreprocessor()
        self.model_loader = ModelLoader(self.base_dir, self.device)
        self.knowledge_loader = KnowledgeLoader(self.base_dir)

        # Modelli e dati (caricati successivamente)
        self.model = None
        self.intent_dict = None

        # Knowledge base
        self.rules = {}
        self.responses = {}

        # Answer management
        self.answer_manager = None
        self.slot_validator = None

        # Slot management (data-driven)
        self.slot_manager = None

        # Rule interpreter (runtime DSL)
        self.rule_interpreter = None

        # Operations
        self.operation_manager = None

        # Conversation balancer
        self.conversation_balancer = None

    def load_models(self) -> bool:
        """
        Carica tutti i modelli ML necessari.

        Returns:
            bool: True se il modello è stato caricato correttamente, False se usa pesi random
        """
        vocab_size = self.model_loader.load_vocab_size()
        self.intent_dict = self.model_loader.load_intent_dict()

        intents_number = len(self.intent_dict)
        self.model, model_loaded = self.model_loader.load_intent_classifier(
            vocab_size,
            intents_number
        )

        return model_loaded
    
    def load_knowledge(self) -> None:
        """Carica rules, responses, conversations e costruisce la lookup table per doping."""
        self.rules, self.responses, conversations = self.knowledge_loader.load_all()

        # Inizializza il ConversationBalancer con i dati delle conversations e l'intent dict
        self.conversation_balancer = ConversationBalancer(conversations, self.intent_dict)

        # Inizializza l'OperationManager con auto-discovery
        self.operation_manager = OperationManager(
            session_manager=self.session_manager,
            entity_manager=self.session_manager.entity_manager,
            auto_discover=True  # Scopre automaticamente tutte le operations
        )

        # Inizializza il RuleInterpreter (nuovo runtime DSL)
        self.rule_interpreter = RuleInterpreter(
            self.rules,
            self.responses,
            operation_manager=self.operation_manager
        )

        # Answer manager legacy (da deprecare gradualmente)
        self.answer_manager = AnswerManager(self.rules)
        self.slot_validator = SlotValidator(self.rules)
        
        # Inizializza il nuovo SlotManager con il RuleInterpreter
        self.slot_manager = SlotManager(self.rules, self.rule_interpreter)

        self.knowledge_loader.build_doping_lookup_table(self.doping_preprocessor)

    def get_response(self, intent_name: str, slots: dict = None) -> tuple[str, str | None, dict]:
        """
        Ottiene una risposta per l'intent specificato usando il RuleInterpreter.

        Args:
            intent_name: Nome dell'intent
            slots: Dizionario degli slot disponibili

        Returns:
            tuple: (risposta, slot_da_attendere, slot_da_impostare_dal_bot)
        """
        if slots is None:
            slots = {}
        
        return self.rule_interpreter.handle_intent_with_bot_slots(intent_name, slots)

    def predict(self, text: str, history: list = None) -> dict:
        """
        Predice intent ed entità per il testo fornito.

        Args:
            text: Testo dell'utente
            history: Storico della conversazione (lista di messaggi)

        Returns:
            dict: {
                'intent': nome_intent,
                'confidence': probabilità,
                'entities': lista_entità,
                'doped': bool,
                'intent_logits': logits grezzi,
                'intent_probs': probabilità per tutti gli intent
            }
        """
        # Doping del testo se attivo
        if DOPING_ACTIVE:
            doped_text = self.doping_preprocessor.dope_input(text)
            text_to_predict = doped_text
            is_doped = doped_text != text
        else:
            text_to_predict = text
            is_doped = False
        
        # Predizione
        result = self.model.predict(text_to_predict)
        intent_idx = result['intent_idx']
        intent_name = self.intent_dict[str(intent_idx)]
        confidence = result['intent_confidence']

        # ConversationBalancer: quando i logits sono incerti (il secondo logit ha una
        # percentuale alta), confronta lo storico con i pattern YAML e boosta il logit
        # dell'intent atteso nel passo successivo del pattern.
        # Se il balancer modifica i logits, ricalcola intent_idx, intent_name e confidence.
        logits = result.get('intent_logits', [])
        intent_probs = result.get('intent_probs', [])
        if self.conversation_balancer is not None:
            balanced_logits = self.conversation_balancer.balance(logits, history or [])
            if balanced_logits is not logits:
                max_logit = max(balanced_logits)
                exp_logits = [math.exp(l - max_logit) for l in balanced_logits]
                sum_exp = sum(exp_logits)
                intent_probs = [e / sum_exp for e in exp_logits]
                intent_idx = intent_probs.index(max(intent_probs))
                intent_name = self.intent_dict[str(intent_idx)]
                confidence = intent_probs[intent_idx]
                logits = balanced_logits
        
        # Fallback per bassa confidenza
        if confidence < MIN_INTENT_CONFIDENCE:
            intent_name = 'low_confidence_fallback'

        return {
            'intent': intent_name,
            'confidence': confidence,
            'entities': result.get('entities', []),
            'doped': is_doped,
            'intent_logits': logits,
            'intent_probs': intent_probs
        }
    
    def chat(self) -> None:
        """
        Avvia l'interfaccia di conversazione testuale.

        Delega la gestione del loop di conversazione al ConversationHandler.
        """
        from agent.conversation_handler import ConversationHandler

        conversation_handler = ConversationHandler(self)
        conversation_handler.run()


def main():
    """Entry point per l'esecuzione standalone dell'agent."""
    agent = Agent()
    
    agent.load_models()
    agent.load_knowledge()
    
    agent.chat()


if __name__ == "__main__":
    main()
