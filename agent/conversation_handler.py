"""
Gestisce l'interfaccia di conversazione con l'utente.
"""
from typing import TYPE_CHECKING

from agent.turn_processor import TurnProcessor, TurnResult

if TYPE_CHECKING:
    from agent.agent import Agent


class ConversationHandler:
    """Gestisce il loop di conversazione e l'interazione con l'utente (I/O da terminale).

    La logica di stato (cancel/inputable/predizione) vive in TurnProcessor,
    condivisa con l'endpoint HTTP (api/chatbot.py); questa classe si occupa solo
    di leggere input, stampare output e il debug verboso della modalità testuale.
    """

    EXIT_COMMANDS = {'esci', 'exit', 'quit', 'q'}

    def __init__(self, agent: 'Agent'):
        self.agent = agent
        self.turn_processor = TurnProcessor(agent)

    def print_header(self, session_id: str, active_sessions_count: int) -> None:
        """Stampa l'header della chat."""
        print("\n" + "="*50)
        print("COGNITOR AGENT - Interfaccia Testuale")
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

    def _print_turn_header(self, session) -> None:
        print("\n" + "─" * 60)
        print(f"[PIPELINE] Turno #{len(session.history) // 2 + 1} | Modalità: {session.agent_mode}")
        print("─" * 60)

    def _print_prediction_debug(self, prediction: dict) -> None:
        print(f"\nIntent: {prediction['intent']} ({prediction['confidence']:.1%})")
        if prediction['entities']:
            entities_str = ', '.join(
                [f"{e['value']} [tipo={e.get('entity', '?')} conf={e.get('confidence', 0):.2f}]"
                 for e in prediction['entities']]
            )
        else:
            entities_str = "nessuna"
        print(f"Entita: {entities_str}")

        intent_probs = prediction.get('intent_probs')
        if intent_probs:
            intent_logits = prediction.get('intent_logits', [])

            sorted_indices = sorted(
                range(len(intent_probs)),
                key=lambda i: intent_probs[i],
                reverse=True
            )

            # Mostra solo intent con probabilità significativa (> 0.0001), max 5
            significant_intents = [
                idx for idx in sorted_indices
                if intent_probs[idx] > 0.0001
            ][:5]

            if significant_intents:
                print(f"\n[DEBUG] Top {len(significant_intents)} Intent (prob > 0.0001):")
                for rank, idx in enumerate(significant_intents, 1):
                    intent_name = self.agent.intent_dict.get(str(idx), f"unknown_{idx}")
                    prob = intent_probs[idx]
                    logit = intent_logits[idx] if idx < len(intent_logits) else 0.0
                    print(f"  {rank}. {intent_name}: {prob:.4f} (logit: {logit:.4f})")
                print()

    def _print_options(self, options: list | None) -> None:
        """Stampa le opzioni (bottoni) offerte per lo slot in attesa, se presenti."""
        if not options:
            return
        choices = "  ".join(f"[{i}] {opt.get('label', opt.get('value'))}" for i, opt in enumerate(options, 1))
        print(f"Opzioni: {choices}\n")

    def _print_result(self, result: TurnResult, session) -> None:
        # Il debug [INPUTABLE]/[PIPELINE]/[SlotManager] è già stampato da TurnProcessor
        # (e da Agent.get_response) nell'ordine cronologico in cui viene generato;
        # qui resta solo la formattazione di presentazione della console.
        print(f"\nCOGNITOR: {result.response}\n")
        self._print_options(result.options)
        if result.kind == "prediction":
            print(f"Cronologia: {len(session.history)} messaggi | Contesto: {session.context}")

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

            if self.handle_exit_command(user_input):
                break

            if not user_input:
                continue

            is_prediction_turn = not (session.agent_mode == "inputable" and session.waiting_for_slot)
            if is_prediction_turn:
                self._print_turn_header(session)

            result = self.turn_processor.process(
                user_input, session, on_predict=self._print_prediction_debug
            )
            self._print_result(result, session)
