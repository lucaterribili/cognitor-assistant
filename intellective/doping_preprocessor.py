import hashlib
import json
import re


STOPWORDS = {
    "ho", "mi", "per", "di", "che", "un", "una", "il", "la", "lo",
    "i", "le", "gli", "e", "a", "in", "con", "su", "da", "del",
    "della", "dei", "delle", "degli", "al", "alla", "ai", "alle",
    "nel", "nella", "nei", "nelle", "si", "ci", "ne", "non", "è"
}


class DopingPreprocessor:
    def __init__(self, short_token_limit: int = 2, avg_token_threshold: int = 3):
        self.short_token_limit = short_token_limit
        self.avg_token_threshold = avg_token_threshold
        self.lookup_table: dict[str, set[str]] = {}

    def _clean_example(self, text: str) -> str:
        return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text).lower().strip()

    def _should_dope(self, examples: list[str]) -> bool:
        cleaned = [self._clean_example(example) for example in examples]
        average_tokens = sum(len(example.split()) for example in cleaned) / len(cleaned)
        return average_tokens <= self.avg_token_threshold

    def build_lookup_table(self, nlu_data: dict):
        for intent_data in nlu_data["nlu"]["intents"]:
            intent_name = intent_data["intent"]
            examples = intent_data["examples"]

            if not self._should_dope(examples):
                continue

            self.lookup_table[intent_name] = set()
            for example in examples:
                clean = self._clean_example(example)
                tokens = clean.split()
                if len(tokens) <= self.short_token_limit:
                    for token in tokens:
                        if token not in STOPWORDS:
                            self.lookup_table[intent_name].add(token)

    def lookup_match(self, text: str) -> str | None:
        tokens = text.lower().strip().split()
        for intent_name, keywords in self.lookup_table.items():
            for token in tokens:
                if token in keywords:
                    return intent_name
        return None

    def _make_example_id(self, intent_name: str, text: str) -> str:
        safe_text = text.replace(" ", "_")
        return f"{intent_name}_{safe_text}"

    def _make_group_id(self, intent_name: str, example: str) -> str:
        """
        Genera un group_id stabile per l'esempio SORGENTE originale (prima di
        normalizzazione/doping). Tutte le varianti derivate dallo stesso
        esempio sorgente (clean_text, normalizer.normalize(...), variante
        "dopata") condividono lo stesso group_id, cosi' il train/val split
        puo' tenerle sempre dalla stessa parte ed evitare data leakage.
        """
        key = f"{intent_name}|{example}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()[:12]

    def get_examples(self, nlu_data: dict) -> list[dict]:
        """
        Restituisce gli esempi SENZA pulirli - mantiene le annotazioni NER
        """
        return [
            {
                "text": ex,
                "intent": intent_data["intent"],
                "group_id": self._make_group_id(intent_data["intent"], ex),
            }
            for intent_data in nlu_data["nlu"]["intents"]
            for ex in intent_data["examples"]
        ]

    def process_dataset(self, nlu_data: dict) -> list[dict]:
        """
        Processa il dataset mantenendo le annotazioni NER nel testo originale.
        Il doping viene applicato solo agli esempi duplicati, non al testo base.
        """
        dataset = []
        for intent_data in nlu_data["nlu"]["intents"]:
            intent_name = intent_data["intent"]
            examples = intent_data["examples"]
            dope = self._should_dope(examples)

            for example in examples:
                group_id = self._make_group_id(intent_name, example)

                # Mantieni l'esempio originale con annotazioni NER
                dataset.append({"text": example, "intent": intent_name, "group_id": group_id})

                # Aggiungi versione "dopata" solo se necessario
                # La versione dopata NON ha annotazioni NER (usa clean)
                if dope:
                    clean = self._clean_example(example)
                    tokens = clean.split()
                    if len(tokens) <= self.short_token_limit:
                        example_id = self._make_example_id(intent_name, clean)
                        prefixed = f"{intent_name} {example_id} {clean}"
                        # Stessa group_id dell'esempio sorgente: e' una variante
                        # near-duplicate, non un esempio indipendente.
                        dataset.append({"text": prefixed, "intent": intent_name, "group_id": group_id})

        return dataset

    def dope_input(self, text: str) -> str:
        tokens = text.strip().split()
        if len(tokens) <= self.short_token_limit:
            matched_intent = self.lookup_match(text)
            if matched_intent:
                example_id = self._make_example_id(matched_intent, text)
                return f"{matched_intent} {example_id} {text}"
        return text
