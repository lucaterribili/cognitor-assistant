"""
Parser per estrarre istruzioni di slot inline dalle risposte.

Supporta la sintassi {SLOT_NAME=value} nelle risposte.
"""
import re
from typing import Tuple, Dict


class ResponseSlotParser:
    """
    Estrae slot inline dalle risposte e li rimuove dal testo visibile.

    Esempio:
        "Imposto Roma {LOCATION=Roma}. Confermi?"
        -> testo_visibile: "Imposto Roma . Confermi?"
        -> slots: {"LOCATION": "Roma"}
    """

    PATTERN = re.compile(r'\{([^=]+)=([^}]+)}')

    @classmethod
    def parse(cls, response: str) -> Tuple[str, Dict]:
        """
        Estrae slot inline dalla risposta.

        Args:
            response: Risposta con potenziali slot inline

        Returns:
            tuple: (risposta_pulita, dict_slot_da_impostare)
        """
        slots = {}
        cleaned_response = response

        for match in cls.PATTERN.finditer(response):
            slot_name = match.group(1).strip()
            slot_value = match.group(2).strip()
            slots[slot_name] = slot_value

        cleaned_response = cls.PATTERN.sub('', response)

        cleaned_response = cleaned_response.strip()

        return cleaned_response, slots

    @classmethod
    def has_inline_slots(cls, response: str) -> bool:
        """Verifica se la risposta contiene slot inline."""
        return bool(cls.PATTERN.search(response))

    @classmethod
    def extract_all_from_responses(cls, responses: list) -> Dict:
        """
        Estrae tutti gli slot definiti in una lista di risposte.

        Args:
            responses: Lista di risposte

        Returns:
            Dict di {slot_name: [valori possibili]}
        """
        all_slots = {}
        for response in responses:
            _, slots = cls.parse(response)
            for slot_name, slot_value in slots.items():
                if slot_name not in all_slots:
                    all_slots[slot_name] = []
                if slot_value not in all_slots[slot_name]:
                    all_slots[slot_name].append(slot_value)
        return all_slots
