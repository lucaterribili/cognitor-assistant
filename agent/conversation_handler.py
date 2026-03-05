"""
Gestisce l'interfaccia di conversazione con l'utente.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import Agent


class ConversationHandler:
    """Gestisce il loop di conversazione e l'interazione con l'utente."""

    EXIT_COMMANDS = {'esci', 'exit', 'quit', 'q'}
    CANCEL_COMMANDS = {'#exit', '#annulla', '#cancel', '#abort'}

    def __init__(self, agent: 'Agent'):
        self.agent = agent

    def print_header(self, session_id: str, active_sessions_count: int) -> None:
        """Stampa l'header della chat."""
        print("\n" + "="*50)
        print("ARIANNA AGENT - Interfaccia Testuale")
        print("="*50)
        print(f"Session ID: {session_id}")
        print(f"Sessioni attive: {active_sessions_count}")
        print("Scrivi un messaggio (o 'esci' per terminare)\n")

    def get_mode_indicator(self, session) -> str:
        """Restituisce l'indicatore della modalità corrente."""
        if session.agent_mode != "predictable":
            return f"[{session.agent_mode.upper()}] "
        return ""

    def handle_exit_command(self, user_input: str) -> bool:
        """
        Gestisce i comandi di uscita.

        Returns:
            True se l'utente vuole uscire, False altrimenti
        """
        if user_input.lower() in self.EXIT_COMMANDS:
            print("\nArrivederci!")
            return True
        return False

    def handle_cancel_command(self, user_input: str, session) -> bool:
        """
        Gestisce i comandi di cancellazione durante input mode.

        Returns:
            True se l'input è stato cancellato, False altrimenti
        """
        if user_input.lower() in self.CANCEL_COMMANDS:
            print("\nArianna: Input annullato. Puoi fornire un nuovo comando.\n")
            session.waiting_for_slot = None
            session.agent_mode = "predictable"
            session.add_message("user", user_input)
            session.add_message("assistant", "Input annullato.", None)
            return True
        return False

    def handle_slot_input(self, user_input: str, session) -> None:
        """Gestisce l'input di uno slot durante la modalità inputable."""
        slot_name = session.waiting_for_slot["slot"]
        pending_intent = session.waiting_for_slot["intent"]

        # Valida l'input usando lo SlotManager (data-driven)
        if not self.agent.slot_manager.validate_slot_value(pending_intent, slot_name, user_input):
            print("\nArianna: Selezione non valida. Riprova.\n")
            session.add_message("user", user_input)
            return

        # Aggiorna il contesto usando SlotManager per consistenza
        session.update_context(slot_name, user_input)
        session.update_context(f"{slot_name}_UNSUPPORTED", False)
        session.waiting_for_slot = None
        session.agent_mode = "predictable"

        # Genera risposta
        response, wait_for_slot, bot_slots = self.agent.get_response(
            pending_intent, session.context
        )

        if bot_slots:
            self._apply_bot_slots(session, bot_slots)

        print(f"\nArianna: {response}\n")

        # Se necessario, attende un altro slot
        if wait_for_slot:
            session.waiting_for_slot = {"intent": pending_intent, "slot": wait_for_slot}

        session.add_message("user", user_input)
        session.add_message("assistant", response, pending_intent)

    def handle_prediction(self, user_input: str, session) -> None:
        """Gestisce la predizione e risposta normale."""
        prediction = self.agent.predict(user_input)

        print(f"\nIntent: {prediction['intent']} ({prediction['confidence']:.1%})")
        entities_str = ', '.join([e['value'] for e in prediction['entities']]) or "nessuna"
        print(f"Entita: {entities_str}")

        self._handle_location_update(user_input, session, prediction)

        response, wait_for_slot, bot_slots = self.agent.get_response(
            prediction['intent'], session.context
        )

        if bot_slots:
            self._apply_bot_slots(session, bot_slots)

        print(f"\nArianna: {response}\n")

        if wait_for_slot:
            session.waiting_for_slot = {"intent": prediction['intent'], "slot": wait_for_slot}
            session.agent_mode = "inputable"

        session.add_message("user", user_input, prediction['intent'], prediction.get('entities', []))
        session.add_message("assistant", response, prediction['intent'])

        print(f"Cronologia: {len(session.history)} messaggi | Contesto: {session.context}")

    def _apply_bot_slots(self, session, bot_slots: dict) -> None:
        """
        Applica gli slot impostati dal bot al contesto della sessione.

        Args:
            session: Sessione corrente
            bot_slots: Dizionario degli slot da impostare
        """
        for slot_name, slot_value in bot_slots.items():
            if slot_value:
                session.update_context(slot_name, slot_value)
                session.update_context(f"{slot_name}_UNSUPPORTED", False)
                print(f"[BotSlot] Impostato {slot_name} = {slot_value}")

    def _handle_location_update(self, user_input: str, session, prediction: dict) -> None:
        """
        Gestisce l'aggiornamento degli slot nel contesto usando il nuovo SlotManager.
        Questo metodo è ora completamente data-driven basato sulle rules.
        """
        # Usa il nuovo SlotManager per aggiornare tutti gli slot rilevanti
        self.agent.slot_manager.update_session_from_prediction(
            session=session,
            current_intent=prediction['intent'],
            entities=prediction.get('entities', []),
            user_input=user_input
        )

    def run(self) -> None:
        """Avvia il loop di conversazione."""
        session_id = self.agent.session_manager.create_session()
        session = self.agent.session_manager.get_session(session_id)

        self.print_header(session_id, len(self.agent.session_manager.get_active_sessions()))

        while True:
            mode_indicator = self.get_mode_indicator(session)
            try:
                user_input = input(f"Tu: {mode_indicator}").strip()
            except EOFError:
                break

            # Check comandi di uscita
            if self.handle_exit_command(user_input):
                break

            # Ignora input vuoti
            if not user_input:
                continue

            # Gestione modalità inputable
            if session.agent_mode == "inputable" and session.waiting_for_slot:
                # Check comandi di cancellazione
                if self.handle_cancel_command(user_input, session):
                    continue

                self.handle_slot_input(user_input, session)
                continue

            # Modalità predictable normale
            self.handle_prediction(user_input, session)

