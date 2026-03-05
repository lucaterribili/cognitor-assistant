import json
import os
import re
import yaml
from pathlib import Path


class NERTagBuilder:

    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, ".cognitor", "ner_tag_builder.json")

        # Se il file non esiste, crealo
        if not os.path.exists(config_path):
            print(f"⚠ File {config_path} non trovato, creazione automatica...")
            self._initialize_from_yaml(config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.tag2id = json.load(f)
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.num_tags = len(self.tag2id)
        self.ENTITY_TYPES = self._extract_entity_types()

    @staticmethod
    def _initialize_from_yaml(config_path: str):
        """
        Scansiona i file YAML degli intents per estrarre tutti i tipi di entità
        e crea il file ner_tag_builder.json
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        intents_dirs = [
            os.path.join(base_dir, 'knowledge', 'intents'),
            os.path.join(base_dir, 'training_data', 'intents')
        ]

        entity_types = set()
        pattern = re.compile(r'\[([^]]+)]\(([^)]+)\)')

        for intents_dir in intents_dirs:
            intents_path = Path(intents_dir)
            if not intents_path.exists():
                continue

            for yaml_file in intents_path.glob('*.yaml'):
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and 'nlu' in data and 'intents' in data['nlu']:
                        for intent in data['nlu']['intents']:
                            if 'examples' in intent:
                                examples = intent['examples']
                                if isinstance(examples, str):
                                    examples = [ex.strip() for ex in examples.strip().split('\n') if ex.strip()]
                                for example in examples:
                                    # Estrae tutti i tipi di entità da [testo](TIPO)
                                    matches = pattern.findall(example)
                                    for _, entity_type in matches:
                                        entity_types.add(entity_type)

        # Costruisce tag2id: O, poi B-* e I-* per ogni tipo di entità
        tag2id = {"O": 0}
        tag_id = 1
        for entity_type in sorted(entity_types):
            tag2id[f"B-{entity_type}"] = tag_id
            tag_id += 1
            tag2id[f"I-{entity_type}"] = tag_id
            tag_id += 1

        # Salva il file
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tag2id, f, ensure_ascii=False, indent=4)

        print(f"✓ Creato {config_path} con {len(entity_types)} tipi di entità: {sorted(entity_types)}")

    def _extract_entity_types(self) -> list[str]:
        """Estrae i tipi di entità dai tag BIO presenti nel file."""
        entity_types = set()
        for tag in self.tag2id.keys():
            if tag.startswith("B-") or tag.startswith("I-"):
                entity_types.add(tag[2:])
        return sorted(list(entity_types))

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
        instance = cls.__new__(cls)
        with open(path, 'r', encoding='utf-8') as f:
            instance.tag2id = json.load(f)
        instance.id2tag = {v: k for k, v in instance.tag2id.items()}
        instance.num_tags = len(instance.tag2id)
        instance.ENTITY_TYPES = instance._extract_entity_types()
        return instance
