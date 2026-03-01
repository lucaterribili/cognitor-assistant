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

    def process_dataset(self, nlu_data: dict) -> list[dict]:
        dataset = []
        for intent_data in nlu_data["nlu"]["intents"]:
            intent_name = intent_data["intent"]
            examples = intent_data["examples"]
            dope = self._should_dope(examples)

            for example in examples:
                clean = self._clean_example(example)
                tokens = clean.split()

                dataset.append({"text": clean, "intent": intent_name})

                if dope and len(tokens) <= self.short_token_limit:
                    example_id = self._make_example_id(intent_name, clean)
                    prefixed = f"{intent_name} {example_id} {clean}"
                    dataset.append({"text": prefixed, "intent": intent_name})

        return dataset

    def dope_input(self, text: str) -> str:
        tokens = text.strip().split()
        if len(tokens) <= self.short_token_limit:
            matched_intent = self.lookup_match(text)
            if matched_intent:
                example_id = self._make_example_id(matched_intent, text)
                return f"{matched_intent} {example_id} {text}"
        return text
