import os.path
import re
from config import BASE_DIR
from classes.simple_tokenizer import SimpleTokenizer


class IntentNormalizer:
    def __init__(self):
        fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
        self.tokenizer = SimpleTokenizer(fasttext_model_path)
        self.entity_pattern = re.compile(r'\[(.*?)]\((.*?)\)')
        self.multiple_spaces_pattern = re.compile(r'\s+')
        self.stop_words = {'di', 'a', 'da', 'in', 'con', 'per', 'su', 'tra', 'fra',
                           'il', 'lo', 'la', 'le', 'gli', 'i', 'un', 'una', 'uno',
                           'e', 'o', 'ma', 'non', 'è', 'sono', 'si', 'che', 'chi',
                           'cosa', 'quale', 'quali', 'qual è', 'qual è'}
        self.important_words = {'come', 'quali', 'quale', 'dove', 'quando', 'perché', 'chi'}

    def extract_entities(self, text):
        entities = []
        positions = []

        for match in self.entity_pattern.finditer(text):
            entity_value = match.group(1)
            entity_type = match.group(2)
            start_pos = match.start()
            end_pos = match.end()

            entities.append((entity_value, entity_type))
            positions.append((start_pos, end_pos))

        return entities, positions

    def split_text_with_entities(self, text, positions):
        parts = []
        last_end = 0

        for (start, end) in positions:
            if start > last_end:
                parts.append(("text", text[last_end:start]))
            entity_index = len([p for p in parts if p[0] == "entity"])
            parts.append(("entity", entity_index))
            last_end = end

        if last_end < len(text):
            parts.append(("text", text[last_end:]))

        return parts

    def normalize_text(self, text):
        tokens = self.tokenizer(text)
        normalized_tokens = [token.lower() for token in tokens if token.lower() not in self.stop_words]
        return ' '.join(normalized_tokens)

    def replace_entity_placeholders(self, text, entities):
        for i, (value, type_) in enumerate(entities):
            text = text.replace(f"__ENTITY_{i}__", f"[{value}]({type_})")
        return text

    def clean_multiple_spaces(self, text):
        return self.multiple_spaces_pattern.sub(' ', text).strip()

    def normalize(self, text):
        entities, positions = self.extract_entities(text)
        parts = self.split_text_with_entities(text, positions)

        normalized_parts = []

        for part_type, part_value in parts:
            if part_type == "entity":
                normalized_parts.append(f"__ENTITY_{part_value}__")
            else:
                normalized_text = self.normalize_text(part_value)
                if normalized_text:
                    normalized_parts.append(normalized_text)

        normalized_text = ' '.join(normalized_parts)
        normalized_text = self.replace_entity_placeholders(normalized_text, entities)
        normalized_text = self.clean_multiple_spaces(normalized_text)

        return normalized_text
