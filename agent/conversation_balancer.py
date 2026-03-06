"""
ConversationBalancer: bilancia i logits del modello in base allo storico della conversazione.
Viene inserito tra la fase di predizione del modello e l'output.
"""
import math


class ConversationBalancer:
    """
    Bilancia i logits del modello in base allo storico della conversazione.

    Viene inserito tra la fase di predizione del modello e l'output.
    Riceve i logits dal modello e ha accesso allo storico della conversazione.

    Quando i logits sono incerti (il secondo logit ha una percentuale alta rispetto
    al primo), confronta lo storico della conversazione con i pattern definiti nello
    YAML per determinare l'intent atteso e boostarlo.
    """

    # Se prob[1] >= UNCERTAINTY_RATIO_THRESHOLD * prob[0], i logits sono considerati incerti
    UNCERTAINTY_RATIO_THRESHOLD = 0.5

    # Numero massimo di turni utente recenti da considerare per il matching
    HISTORY_WINDOW = 5

    # Quantità aggiunta al logit dell'intent suggerito dal pattern
    BOOST_AMOUNT = 2.0

    def __init__(self, conversations: dict = None, intent_dict: dict = None):
        """
        Args:
            conversations: Dizionario delle conversations caricate dai file YAML.
            intent_dict: Dizionario {str(idx): intent_name} per mappare indici a nomi.
        """
        self.conversations = conversations or {}
        self.intent_dict = intent_dict or {}

        # Reverse mapping: intent_name -> indice intero
        self._intent_name_to_idx = {v: int(k) for k, v in self.intent_dict.items()}

        # Pre-calcola le sequenze di intent utente per ogni pattern di conversazione
        self._pattern_sequences = self._build_pattern_sequences()

    def _build_pattern_sequences(self) -> list:
        """
        Costruisce le sequenze di intent utente da tutti i pattern di conversazione.

        Returns:
            list[list[str]]: Lista di sequenze, una per ogni flusso di conversazione.
        """
        sequences = []
        for flow_data in self.conversations.values():
            steps = flow_data.get('steps', [])
            user_intents = [step['user'] for step in steps if 'user' in step]
            if user_intents:
                sequences.append(user_intents)
        return sequences

    @staticmethod
    def _softmax(logits: list) -> list:
        """Calcola le probabilità softmax dai logits."""
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        return [e / sum_exp for e in exp_logits]

    def _is_uncertain(self, probs: list) -> bool:
        """
        Controlla se i logits sono incerti.

        I logits sono incerti quando la seconda probabilità più alta è >= UNCERTAINTY_RATIO_THRESHOLD
        rispetto alla prima probabilità più alta.
        """
        if len(probs) < 2:
            return False
        sorted_probs = sorted(probs, reverse=True)
        return sorted_probs[1] >= self.UNCERTAINTY_RATIO_THRESHOLD * sorted_probs[0]

    def _get_recent_user_intents(self, history: list) -> list:
        """
        Estrae gli intent utente recenti dallo storico della conversazione.

        Args:
            history: Lista di messaggi nella forma {role, content, intent, ...}.

        Returns:
            list[str]: Sequenza degli ultimi intent utente (al massimo HISTORY_WINDOW).
        """
        user_intents = [
            msg['intent']
            for msg in history
            if msg.get('role') == 'user' and msg.get('intent')
        ]
        return user_intents[-self.HISTORY_WINDOW:]

    def _find_next_expected_intent(self, recent_intents: list) -> str | None:
        """
        Trova il prossimo intent atteso confrontando la cronologia con i pattern YAML.

        Scorre tutti i pattern di conversazione e cerca la corrispondenza più lunga
        tra la coda degli intent recenti e un sotto-segmento del pattern.
        Se trovata, restituisce l'intent previsto nel passo successivo.

        Args:
            recent_intents: Sequenza di intent utente recenti.

        Returns:
            str | None: Nome dell'intent atteso, oppure None se nessun pattern corrisponde.
        """
        if not recent_intents:
            return None

        best_match_length = 0
        best_next_intent = None

        for pattern in self._pattern_sequences:
            n = len(recent_intents)
            # Prova corrispondenze dalla più lunga alla più corta
            for match_len in range(min(n, len(pattern)), 0, -1):
                if match_len <= best_match_length:
                    # Non può migliorare il match già trovato
                    break
                recent_suffix = recent_intents[-match_len:]
                for i in range(len(pattern) - match_len + 1):
                    if pattern[i:i + match_len] == recent_suffix:
                        next_idx = i + match_len
                        if next_idx < len(pattern):
                            best_match_length = match_len
                            best_next_intent = pattern[next_idx]
                        break

        return best_next_intent

    def balance(self, logits: list, history: list = None) -> list:
        """
        Applica il bilanciamento ai logits in base allo storico della conversazione.

        Se i logits sono incerti e la cronologia corrisponde a un pattern YAML,
        boosta il logit dell'intent atteso nel passo successivo del pattern.

        Args:
            logits: Logits grezzi dal modello.
            history: Storico della conversazione (lista di messaggi).

        Returns:
            list: Logits eventualmente modificati.
        """
        if not logits or not history:
            return logits

        probs = self._softmax(logits)

        if not self._is_uncertain(probs):
            return logits

        recent_intents = self._get_recent_user_intents(history)
        next_intent = self._find_next_expected_intent(recent_intents)

        if next_intent and next_intent in self._intent_name_to_idx:
            idx = self._intent_name_to_idx[next_intent]
            if idx < len(logits):
                logits = list(logits)
                logits[idx] += self.BOOST_AMOUNT

        return logits
