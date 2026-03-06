"""
DialogueStatePolicy: gestisce lo stato della conversazione e predice la prossima azione.

Ispirata alla TED (Transformer Embedding Dialogue) Policy di Rasa:
- TED rappresenta il dialogo come sequenza di features (intent + entità + azioni precedenti)
- Usa un dual encoder (dialogo + azione) con confronto di similarità
- Viene addestrata su "storie" di conversazione

Questa implementazione adatta il principio TED all'architettura del progetto:
- Usa le "stories" (conversations YAML) come Rasa usa le storie
- Rappresenta lo stato come sequenza di intent utente recenti
- Calcola la similarità tramite longest-suffix match (efficiente, nessun training richiesto)
- Predice la prossima azione bot con un confidence score
- Opera a livello di gestione del dialogo (DM), non NLU

A differenza del ConversationBalancer (che modificava i logits NLU), questa policy
determina la risposta direttamente a partire dallo stato dialogico.
"""


class DialogueStatePolicy:
    """
    Policy TED-inspired per la gestione dello stato della conversazione.

    Dato lo storico della conversazione e l'intent corrente dell'utente,
    predice la prossima azione (response key) che il bot dovrebbe eseguire.

    Architettura:
    - Stato = sequenza degli ultimi HISTORY_WINDOW intent utente
    - Transizioni = tabella (contesto, intent_corrente) → prossima_azione
      costruita dalle storie YAML
    - Score = rapporto di corrispondenza del suffisso più lungo
    - Predizione = azione con score massimo se >= MIN_CONFIDENCE
    """

    # Numero massimo di turni utente recenti da considerare come contesto
    HISTORY_WINDOW = 5

    # Confidence minima per applicare la predizione della policy
    MIN_CONFIDENCE = 0.4

    def __init__(self, conversations: dict = None):
        """
        Args:
            conversations: Dizionario delle conversations caricate dai file YAML.
                           Struttura: {flow_name: {'steps': [{'user': ..., 'bot': ...}]}}
        """
        self.conversations = conversations or {}
        self._story_transitions = self._build_story_transitions()

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

    def predict_next_action(self, current_intent: str, history: list) -> dict | None:
        """
        Predice la prossima azione del bot dato l'intent corrente e lo storico.

        Cerca tra tutte le transizioni quelle con user_intent corrispondente
        e seleziona quella il cui contesto ha il punteggio di match più alto.

        Args:
            current_intent: Intent utente appena predetto dal modello NLU.
            history: Storico della conversazione (lista di messaggi).

        Returns:
            dict con 'action' (str) e 'confidence' (float), oppure None se
            nessuna transizione supera MIN_CONFIDENCE.
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
