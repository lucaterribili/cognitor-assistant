"""
DialogueStatePolicy: gestisce lo stato della conversazione e predice la prossima azione.

Implementa una policy di dialogo basata su un modello ML (DialoguePolicy) addestrato
sulla storia delle conversazioni presenti nei file YAML.

Ispirata alla TED (Transformer Embedding Dialogue) Policy di Rasa:
- Addestrata su "storie" di conversazione (conversations YAML)
- Rappresenta lo stato come sequenza di intent utente recenti
- Predice la prossima azione tramite un GRU encoder + classificatore
- Opera a livello di gestione del dialogo (DM), non NLU

Se il modello addestrato non è disponibile (prima esecuzione della pipeline),
ricade su un approccio euristico basato su longest-suffix match per garantire
la retrocompatibilità.
"""

import json
import os

import torch


class DialogueStatePolicy:
    """
    Policy ML per la gestione dello stato della conversazione.

    Usa il modello DialoguePolicy (GRU encoder + classificatore) addestrato
    nella pipeline per predire la prossima azione del bot.

    Fallback: approccio euristico basato su longest-suffix match delle storie YAML,
    usato quando il modello addestrato non è ancora disponibile.
    """

    # Numero massimo di turni utente recenti da considerare come contesto
    HISTORY_WINDOW = 5

    # Confidence minima per applicare la predizione della policy
    MIN_CONFIDENCE = 0.4

    def __init__(self, conversations: dict = None, base_dir: str = None):
        """
        Args:
            conversations: Dizionario delle conversations caricate dai file YAML.
                           Struttura: {flow_name: {'steps': [{'user': ..., 'bot': ...}]}}
            base_dir:      Directory base del progetto. Se None, usa config.BASE_DIR.
        """
        from config import BASE_DIR as _BASE_DIR

        self.conversations = conversations or {}
        _base_dir = base_dir or _BASE_DIR

        # Percorsi per il modello ML e i dizionari
        self._model_path = os.path.join(_base_dir, 'models', 'dialogue_policy.pth')
        self._intent_dict_path = os.path.join(_base_dir, '.cognitor', 'dialogue_intent_dict.json')
        self._action_dict_path = os.path.join(_base_dir, '.cognitor', 'dialogue_action_dict.json')

        # Stato del modello ML
        self._model = None
        self._intent_dict: dict[str, int] = {}    # nome → id (1-indexed)
        self._action_dict: dict[str, int] = {}    # nome → id (1-indexed)
        self._action_dict_inv: dict[int, str] = {}  # id → nome

        # Tenta di caricare il modello ML addestrato
        self._use_ml = False
        self._load_ml_model()

        # Costruisci le transizioni euristiche (usate come fallback)
        self._story_transitions = self._build_story_transitions()

    # ------------------------------------------------------------------ #
    #  Caricamento modello ML                                              #
    # ------------------------------------------------------------------ #

    def _load_ml_model(self) -> None:
        """
        Carica il modello DialoguePolicy addestrato dalla pipeline.

        Se i file del modello non esistono o il caricamento fallisce,
        _use_ml rimane False e si userà l'approccio euristico.
        """
        required = [self._model_path, self._intent_dict_path, self._action_dict_path]
        if not all(os.path.exists(p) for p in required):
            return

        try:
            from intellective.dialogue_policy import (
                DialoguePolicy,
                DIALOGUE_POLICY_EMBED_DIM,
                DIALOGUE_POLICY_HIDDEN_DIM,
                DIALOGUE_POLICY_DROPOUT,
            )

            with open(self._intent_dict_path, 'r', encoding='utf-8') as f:
                self._intent_dict = json.load(f)
            with open(self._action_dict_path, 'r', encoding='utf-8') as f:
                self._action_dict = json.load(f)

            self._action_dict_inv = {v: k for k, v in self._action_dict.items()}

            model = DialoguePolicy(
                num_intents=len(self._intent_dict),
                num_actions=len(self._action_dict),
                embed_dim=DIALOGUE_POLICY_EMBED_DIM,
                hidden_dim=DIALOGUE_POLICY_HIDDEN_DIM,
                dropout=DIALOGUE_POLICY_DROPOUT,
            )
            state_dict = torch.load(self._model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

            self._model = model
            self._use_ml = True
            print("OK Dialogue Policy ML caricata")

        except Exception as e:
            print(f"WARNING Dialogue Policy ML non disponibile: {e}")
            self._use_ml = False

    # ------------------------------------------------------------------ #
    #  Predizione ML                                                       #
    # ------------------------------------------------------------------ #

    def _ml_predict(self, current_intent: str, history: list) -> dict | None:
        """
        Predice la prossima azione usando il modello ML addestrato.

        Args:
            current_intent: Intent utente corrente.
            history:        Storico della conversazione.

        Returns:
            dict con 'action' e 'confidence', oppure None.
        """
        if current_intent not in self._intent_dict:
            return None

        # Estrai la sequenza degli intent utente precedenti
        context_ids = [
            self._intent_dict[intent]
            for intent in self._extract_user_intent_sequence(history)
            if intent in self._intent_dict
        ]

        # Tensori di input (batch size 1)
        if context_ids:
            context_tensor = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0)
        else:
            # Nessun contesto: usa un singolo token di padding
            context_tensor = torch.zeros(1, 1, dtype=torch.long)

        current_tensor = torch.tensor(
            [self._intent_dict[current_intent]], dtype=torch.long
        )

        # Predizione (action_idx è 0-indexed per CrossEntropyLoss)
        action_idx, confidence = self._model.predict(context_tensor, current_tensor)

        # Il dizionario usa id 1-indexed → aggiungi 1
        action_name = self._action_dict_inv.get(action_idx + 1)

        if action_name and confidence >= self.MIN_CONFIDENCE:
            return {'action': action_name, 'confidence': confidence}

        return None

    # ------------------------------------------------------------------ #
    #  Approccio euristico (fallback / backward compatibility)             #
    # ------------------------------------------------------------------ #

    def _build_story_transitions(self) -> list:
        """
        Costruisce la tabella di transizioni dalle storie YAML.

        Per ogni posizione i in ogni storia, genera una transizione:
            context  = sequenza degli intent utente nei passi [0..i-1]
            user     = intent utente al passo i
            action   = azione bot al passo i

        Returns:
            list[dict]: Lista di transizioni con chiavi 'context', 'user_intent', 'next_action'.
        """
        transitions = []
        for flow_data in self.conversations.values():
            steps = flow_data.get('steps', [])

            # Normalizza i passi in coppie (user_intent, bot_action)
            pairs = []
            for step in steps:
                user_intent = step.get('user')
                bot_action = step.get('bot')
                if user_intent is not None:
                    pairs.append((user_intent, bot_action))

            # Genera una transizione per ogni passo con azione bot definita
            for i, (user_intent, bot_action) in enumerate(pairs):
                if not bot_action:
                    continue

                # Il contesto sono gli intent utente dei passi precedenti
                context_start = max(0, i - self.HISTORY_WINDOW)
                context = [u for u, _ in pairs[context_start:i]]

                transitions.append({
                    'context': context,
                    'user_intent': user_intent,
                    'next_action': bot_action,
                })

        return transitions

    def _extract_user_intent_sequence(self, history: list) -> list:
        """
        Estrae la sequenza degli intent utente dallo storico della conversazione.

        Args:
            history: Lista di messaggi {role, content, intent, ...}.

        Returns:
            list[str]: Ultimi HISTORY_WINDOW intent utente (escluso l'ultimo turno corrente).
        """
        user_intents = [
            msg['intent']
            for msg in history
            if msg.get('role') == 'user' and msg.get('intent')
        ]
        return user_intents[-self.HISTORY_WINDOW:]

    def _score_context_match(self, current_context: list, story_context: list) -> float:
        """
        Calcola il punteggio di corrispondenza tra il contesto corrente e quello di una storia.

        Usa il longest-suffix match: premia i contesti che condividono il
        suffisso più lungo con la storia.

        Args:
            current_context: Sequenza di intent utente recenti.
            story_context: Sequenza di intent utente attesi dalla storia.

        Returns:
            float: Punteggio in [0.0, 1.0].
        """
        if not story_context:
            # Transizione senza contesto (primo passo della storia): match sempre
            return 0.5

        if not current_context:
            return 0.0

        # Cerca la corrispondenza del suffisso più lungo
        max_match = 0
        n = min(len(current_context), len(story_context))

        for length in range(n, 0, -1):
            if current_context[-length:] == story_context[-length:]:
                max_match = length
                break

        if max_match == 0:
            return 0.0

        # Punteggio normalizzato rispetto alla lunghezza del contesto story
        return max_match / len(story_context)

    def _heuristic_predict(self, current_intent: str, history: list) -> dict | None:
        """
        Predice la prossima azione usando l'approccio euristico (fallback).

        Cerca tra tutte le transizioni quella con user_intent corrispondente
        e il contesto con lo score longest-suffix più alto.

        Args:
            current_intent: Intent utente corrente.
            history:        Storico della conversazione.

        Returns:
            dict con 'action' e 'confidence', oppure None.
        """
        if not current_intent or not self._story_transitions:
            return None

        current_context = self._extract_user_intent_sequence(history)

        best_action = None
        best_score = 0.0

        for transition in self._story_transitions:
            if transition['user_intent'] != current_intent:
                continue

            score = self._score_context_match(current_context, transition['context'])

            if score > best_score:
                best_score = score
                best_action = transition['next_action']

        if best_action and best_score >= self.MIN_CONFIDENCE:
            return {'action': best_action, 'confidence': best_score}

        return None

    def predict_next_action(self, current_intent: str, history: list) -> dict | None:
        """
        Predice la prossima azione del bot dato l'intent corrente e lo storico.

        Usa il modello ML addestrato se disponibile; altrimenti ricade
        sull'approccio euristico (longest-suffix match sulle storie YAML).

        Args:
            current_intent: Intent utente appena predetto dal modello NLU.
            history:        Storico della conversazione (lista di messaggi).

        Returns:
            dict con 'action' (str) e 'confidence' (float), oppure None se
            nessuna predizione supera MIN_CONFIDENCE.
        """
        if not current_intent:
            return None

        if self._use_ml:
            return self._ml_predict(current_intent, history)

        return self._heuristic_predict(current_intent, history)
