import os.path
import re
from config import BASE_DIR
import spacy


class IntentNormalizer:
    def __init__(self):
        spacy_path = os.path.join(BASE_DIR, 'models', 'nlp_it_helper')
        self.nlp = spacy.load(spacy_path)
        # Pattern per individuare le entità nel formato [valore](TIPO)
        self.entity_pattern = re.compile(r'\[(.*?)]\((.*?)\)')
        # Pattern per rimuovere spazi multipli
        self.multiple_spaces_pattern = re.compile(r'\s+')
        self.setup_stopwords()

    def setup_stopwords(self):
        stop_words = ['Qual', 'il', 'i']
        for stop_word in stop_words:
            self.nlp.vocab[stop_word].is_stop = True
        important_words = ['Come', 'Quali', 'Quale', 'Dove', 'Quando', 'Perché', 'Chi']
        for important_word in important_words:
            self.nlp.vocab[important_word].is_stop = False

    def extract_entities(self, text):
        """
        Estrae le entità dal testo e restituisce le loro informazioni e posizioni.

        Returns:
            tuple: (entities, positions) dove:
                  - entities è una lista di tuple (valore, tipo)
                  - positions è una lista di tuple (posizione_inizio, posizione_fine)
        """
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
        """
        Divide il testo in parti di testo regolare e segnaposto per le entità.

        Returns:
            list: Lista di tuple (tipo, valore) dove tipo è "text" o "entity"
        """
        parts = []
        last_end = 0

        for (start, end) in positions:
            if start > last_end:
                # Aggiungi il testo prima dell'entità
                parts.append(("text", text[last_end:start]))
            # Aggiungi l'entità come un placeholder speciale
            entity_index = len([p for p in parts if p[0] == "entity"])
            parts.append(("entity", entity_index))
            last_end = end

        # Aggiungi l'ultimo pezzo di testo dopo l'ultima entità
        if last_end < len(text):
            parts.append(("text", text[last_end:]))

        return parts

    def normalize_text(self, text):
        """
        Normalizza il testo usando spaCy (lemmatizzazione e rimozione stopwords).

        Returns:
            str: Testo normalizzato
        """
        doc = self.nlp(text)

        normalized_text = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
        return normalized_text

    def replace_entity_placeholders(self, text, entities):
        """
        Sostituisce i segnaposto con le entità originali.

        Returns:
            str: Testo con le entità ripristinate
        """
        for i, (value, type_) in enumerate(entities):
            text = text.replace(f"__ENTITY_{i}__", f"[{value}]({type_})")
        return text

    def clean_multiple_spaces(self, text):
        """
        Rimuove gli spazi multipli dal testo.

        Returns:
            str: Testo con spazi normalizzati
        """
        return self.multiple_spaces_pattern.sub(' ', text).strip()

    def normalize(self, text):
        """
        Normalize the input text using spaCy, preserving entities in [value](TYPE) format.
        """
        # 1. Estrai le entità e le loro posizioni
        entities, positions = self.extract_entities(text)

        # 2. Dividi il testo in parti (testo/entità)
        parts = self.split_text_with_entities(text, positions)

        # 3. Normalizza le parti di testo e mantieni i segnaposti per le entità
        normalized_parts = []

        for part_type, part_value in parts:
            if part_type == "entity":
                # Mantieni il placeholder dell'entità
                normalized_parts.append(f"__ENTITY_{part_value}__")
            else:
                # Normalizza solo il testo
                normalized_text = self.normalize_text(part_value)
                if normalized_text:  # Aggiungi solo se non è vuoto
                    normalized_parts.append(normalized_text)

        # 4. Ricombina e sostituisci i placeholder con le entità originali
        normalized_text = ' '.join(normalized_parts)
        normalized_text = self.replace_entity_placeholders(normalized_text, entities)

        # 5. Rimuovi gli spazi multipli
        normalized_text = self.clean_multiple_spaces(normalized_text)

        return normalized_text