import json


class NERTagBuilder:

    ENTITY_TYPES = ["PERSON", "LOCATION", "DATE", "TIME", "NUMBER", "PRODUCT", "COMMAND", "TOPIC", "EMAIL", "TEAM"]

    def __init__(self):
        self.tag2id = {"O": 0}
        for ent in self.ENTITY_TYPES:
            self.tag2id[f"B-{ent}"] = len(self.tag2id)
            self.tag2id[f"I-{ent}"] = len(self.tag2id)
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.num_tags = len(self.tag2id)

    def align_tokens_to_bio(self, clean_text: str, tokens: list[str], entities: list[dict]) -> list[int]:
        """
        Allinea i BIO tag ai token usando gli offset carattere sul testo pulito.
        Gestisce tokenizzatori che alterano leggermente il testo (lowercase, ecc.)
        """
        # 1. Costruisce mapping char → tag per TUTTI i caratteri dell'entità
        char_tags = ["O"] * len(clean_text)
        for ent in entities:
            s, e, label = ent["start"], ent["end"], ent["entity"]
            if label not in self.ENTITY_TYPES:
                continue
            # Marca TUTTI i caratteri dell'entità, non solo il primo
            for i in range(s, e):
                if i == s:
                    char_tags[i] = f"B-{label}"
                else:
                    char_tags[i] = f"I-{label}"

        # 2. Trova offset esatti dei token nel testo
        token_offsets = []
        search_from = 0
        for token in tokens:
            # Prova prima con case-sensitive
            pos = clean_text.find(token, search_from)
            if pos == -1:
                # Fallback: case-insensitive
                pos = clean_text.lower().find(token.lower(), search_from)

            if pos != -1:
                token_offsets.append((pos, pos + len(token)))
                search_from = pos + len(token)
            else:
                token_offsets.append(None)

        # 3. Assegna tag ai token usando il tag PREDOMINANTE nei suoi caratteri
        tag_ids = []
        for offset in token_offsets:
            if offset is None:
                tag_ids.append(self.tag2id["O"])
                continue

            start, end = offset
            # Raccoglie tutti i tag dei caratteri che compongono il token
            token_char_tags = [char_tags[i] for i in range(start, min(end, len(char_tags)))]

            # Priorità: B-* > I-* > O
            # Prende il primo B- se presente
            b_tags = [t for t in token_char_tags if t.startswith("B-")]
            if b_tags:
                tag = b_tags[0]
            else:
                # Altrimenti prende il primo I- se presente
                i_tags = [t for t in token_char_tags if t.startswith("I-")]
                if i_tags:
                    tag = i_tags[0]
                else:
                    tag = "O"

            tag_ids.append(self.tag2id.get(tag, 0))

        return tag_ids

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.tag2id, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, path: str):
        instance = cls()
        with open(path, 'r', encoding='utf-8') as f:
            instance.tag2id = json.load(f)
        instance.id2tag = {v: k for k, v in instance.tag2id.items()}
        instance.num_tags = len(instance.tag2id)
        return instance
