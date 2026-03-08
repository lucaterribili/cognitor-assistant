"""
Agent principale per il chatbot Cognitor.
Coordina i modelli ML, la gestione delle sessioni e le risposte.
"""
import os
import random
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
from agent.dialogue_state_policy import DialogueStatePolicy


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

        # Dialogue state policy (TED-inspired)
        self.dialogue_state_policy = None

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

        # Inizializza la DialogueStatePolicy con i dati delle conversations
        self.dialogue_state_policy = DialogueStatePolicy(conversations)

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

    def get_response(self, intent_name: str, slots: dict = None, history: list = None) -> tuple[str, str | None, dict]:
        """
        Ottiene una risposta per l'intent specificato.

        Usa la DialogueStatePolicy per predire la prossima azione in base allo
        storico della conversazione. Se la policy è sufficientemente sicura,
        usa l'azione suggerita come response key; altrimenti usa il RuleInterpreter.

        Args:
            intent_name: Nome dell'intent
            slots: Dizionario degli slot disponibili
            history: Storico della conversazione (opzionale, per la dialogue policy)

        Returns:
            tuple: (risposta, slot_da_attendere, slot_da_impostare_dal_bot)
        """
        if slots is None:
            slots = {}

        _history = history or []
        print(f"\n[PIPELINE] get_response chiamato → intent='{intent_name}' | slots={list(slots.keys())} | history_len={len(_history)}")

        # --- Priorità 1: RuleInterpreter ---
        # Se esiste una rule nel DSL per questo intent, il RuleInterpreter
        # ha sempre la precedenza sulla TED Policy. Questo garantisce che:
        # - slot filling e modalità inputable funzionino sempre
        # - le operations vengano eseguite
        # - la logica di business dichiarativa non venga bypassata
        rule = self.rules.get(intent_name)
        if rule is not None:
            print(f"[PIPELINE] RuleInterpreter PRIORITARIO → rule trovata per '{intent_name}'")
            print("[PIPELINE] Risposta sorgente: RuleInterpreter")
            return self.rule_interpreter.handle_intent_with_bot_slots(intent_name, slots)

        # --- Priorità 2: TED / Dialogue State Policy (solo se nessuna rule) ---
        if not self.dialogue_state_policy:
            print("[PIPELINE] TED Policy SALTATA → dialogue_state_policy non inizializzata")
        else:
            policy_action = self.dialogue_state_policy.predict_next_action(intent_name, _history)
            if policy_action:
                action_key = policy_action['action']
                confidence = policy_action['confidence']
                response_list = self.responses.get(action_key)
                print(f"[PIPELINE] TED Policy ATTIVA → azione='{action_key}' | confidenza={confidence:.3f} | risposte_disponibili={len(response_list) if response_list else 0}")
                if response_list:
                    chosen = random.choice(response_list)
                    print(f"[PIPELINE] Risposta sorgente: TED Policy")
                    return chosen, None, {}
                else:
                    print(f"[PIPELINE] TED Policy → azione '{action_key}' non trovata nelle responses, fallback a RuleInterpreter")
            else:
                print(f"[PIPELINE] TED Policy NON INTERVENUTA → nessuna azione per intent '{intent_name}'")

        # --- Priorità 3: Fallback RuleInterpreter (intent senza rule né TED) ---
        print("[PIPELINE] Risposta sorgente: RuleInterpreter")
        return self.rule_interpreter.handle_intent_with_bot_slots(intent_name, slots)

    def predict(self, text: str) -> dict:
        """
        Predice intent ed entità per il testo fornito.

        Args:
            text: Testo dell'utente

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

        logits = result.get('intent_logits', [])
        intent_probs = result.get('intent_probs', [])

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
