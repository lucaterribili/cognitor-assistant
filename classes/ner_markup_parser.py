import re


class NERMarkupParser:
    """
    Parsa testi con markup tipo [valore](LABEL) ed estrae:
    - testo pulito
    - lista di entità con offset carattere
    """

    PATTERN = re.compile(r'\[([^]]+)]\(([^)]+)\)')

    def parse(self, text: str):
        """
        Input:  "ciao sono [Mario](PERSON) a [Roma](LOCATION)"
        Output: ("ciao sono Mario a Roma", [
                    {"start": 10, "end": 15, "entity": "PERSON",   "value": "Mario"},
                    {"start": 18, "end": 22, "entity": "LOCATION", "value": "Roma"},
                ])
        """
        clean_text = ""
        entities = []
        last_end = 0

        for match in self.PATTERN.finditer(text):
            value = match.group(1)
            label = match.group(2).upper()

            clean_text += text[last_end:match.start()]
            entity_start = len(clean_text)
            clean_text += value
            entity_end = len(clean_text)

            entities.append({
                "start": entity_start,
                "end": entity_end,
                "entity": label,
                "value": value
            })
            last_end = match.end()

        clean_text += text[last_end:]
        return clean_text, entities
