"""
Dataset Validator - Valida intenti ed entità NER nel dataset di training
"""
import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re


class DatasetValidator:
    """Valida il dataset di training per intenti ed entità"""

    def __init__(self, knowledge_path: str, training_path: str = None):
        self.knowledge_path = Path(knowledge_path)
        self.training_path = Path(training_path) if training_path else None
        self.errors = []
        self.warnings = []

    def validate_all(self) -> bool:
        """
        Valida tutto il dataset
        Returns: True se tutto è valido, False altrimenti
        """
        print("=" * 80)
        print("VALIDAZIONE DATASET")
        print("=" * 80)

        # Valida entità NER
        ner_valid = self.validate_ner_entities()

        # Valida intenti
        intents_valid = self.validate_intents()

        # Report finale
        self._print_report()

        return ner_valid and intents_valid and len(self.errors) == 0

    def validate_ner_entities(self) -> bool:
        """Valida che non ci siano entità NER duplicate (case-insensitive)"""
        print("\n[1] VALIDAZIONE ENTITÀ NER")
        print("-" * 80)

        entities = defaultdict(list)  # {entity_lower: [occorrenze con case diverso]}
        entity_regex = re.compile(r'\[(.*?)\]\(([A-Za-z_]+)\)')

        # Raccoglie tutte le entità da tutti i file YAML
        yaml_files = self._get_all_yaml_files()

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if not data or 'nlu' not in data:
                    continue

                intents_data = data['nlu'].get('intents', [])

                for intent_obj in intents_data:
                    examples = intent_obj.get('examples', [])

                    for example in examples:
                        # Trova tutte le entità nel formato [testo](ENTITY)
                        matches = entity_regex.findall(example)

                        for text, entity_type in matches:
                            entity_lower = entity_type.lower()

                            # Traccia ogni variazione di case
                            if entity_type not in entities[entity_lower]:
                                entities[entity_lower].append(entity_type)

            except Exception as e:
                self.warnings.append(f"Errore lettura {yaml_file}: {str(e)}")

        # Verifica duplicati case-insensitive
        has_errors = False

        for entity_lower, variations in entities.items():
            if len(variations) > 1:
                has_errors = True
                error_msg = f"❌ ENTITÀ CON CASE DIVERSO: {variations}"
                self.errors.append(error_msg)
                print(f"  {error_msg}")
                print(f"     → Normalizza tutte le occorrenze a: {variations[0]}")

        if not has_errors:
            unique_entities = set()
            for variations in entities.values():
                unique_entities.add(variations[0])
            print(f"  ✓ Nessun duplicato case-insensitive trovato")
            print(f"  ✓ Entità uniche: {len(unique_entities)}")
            print(f"     {sorted(unique_entities)}")

        return not has_errors

    def validate_intents(self) -> bool:
        """Valida intenti: cerca duplicati ed esempi presenti in intenti diversi"""
        print("\n[2] VALIDAZIONE INTENTI")
        print("-" * 80)

        intent_names = defaultdict(list)  # {intent_name: [files]}
        examples_to_intents = defaultdict(list)  # {example: [intents]}

        yaml_files = self._get_all_yaml_files()

        # Raccoglie tutti gli intenti ed esempi
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if not data or 'nlu' not in data:
                    continue

                intents_data = data['nlu'].get('intents', [])

                for intent_obj in intents_data:
                    intent_name = intent_obj.get('intent')
                    examples = intent_obj.get('examples', [])

                    if not intent_name:
                        continue

                    # Traccia dove compare ogni intent
                    intent_names[intent_name].append(str(yaml_file))

                    # Traccia a quali intenti appartiene ogni esempio
                    for example in examples:
                        # Normalizza l'esempio rimuovendo le annotazioni NER
                        normalized = self._normalize_example(example)
                        examples_to_intents[normalized].append(intent_name)

            except Exception as e:
                self.warnings.append(f"Errore lettura {yaml_file}: {str(e)}")

        # Valida intenti duplicati
        has_duplicate_intents = False
        print("\n  [2.1] Intenti Duplicati:")

        for intent_name, files in intent_names.items():
            if len(files) > 1:
                has_duplicate_intents = True
                error_msg = f"❌ INTENT DUPLICATO: '{intent_name}' presente in {len(files)} file"
                self.errors.append(error_msg)
                print(f"    {error_msg}")
                for file in files:
                    print(f"       - {file}")

        if not has_duplicate_intents:
            print(f"    ✓ Nessun intent duplicato")
            print(f"    ✓ Intenti totali: {len(intent_names)}")

        # Valida esempi duplicati tra intenti
        has_duplicate_examples = False
        print("\n  [2.2] Esempi Presenti in Intenti Diversi:")

        for example, intents in examples_to_intents.items():
            if len(intents) > 1:
                # Rimuove duplicati mantenendo l'ordine
                unique_intents = []
                for intent in intents:
                    if intent not in unique_intents:
                        unique_intents.append(intent)

                if len(unique_intents) > 1:
                    has_duplicate_examples = True
                    error_msg = f"❌ ESEMPIO DUPLICATO: '{example}'"
                    self.errors.append(error_msg)
                    print(f"    {error_msg}")
                    print(f"       Presente in: {unique_intents}")

        if not has_duplicate_examples:
            print(f"    ✓ Nessun esempio duplicato tra intenti")
            print(f"    ✓ Esempi totali: {len(examples_to_intents)}")

        return not has_duplicate_intents and not has_duplicate_examples

    def _normalize_example(self, example: str) -> str:
        """Rimuove le annotazioni NER da un esempio per confrontarlo"""
        # Rimuove pattern [testo](ENTITY) e lascia solo il testo
        normalized = re.sub(r'\[(.*?)\]\([A-Za-z_]+\)', r'\1', example)
        # Normalizza spazi
        normalized = ' '.join(normalized.split())
        return normalized.strip().lower()

    def _get_all_yaml_files(self) -> List[Path]:
        """Raccoglie tutti i file YAML da knowledge e training_data"""
        yaml_files = []

        # File da knowledge/intents
        knowledge_intents = self.knowledge_path / 'intents'
        if knowledge_intents.exists():
            yaml_files.extend(knowledge_intents.glob('*.yaml'))

        # File da training_data/intents
        if self.training_path and self.training_path.exists():
            training_intents = self.training_path / 'intents'
            if training_intents.exists():
                yaml_files.extend(training_intents.glob('*.yaml'))

        return yaml_files

    def _print_report(self):
        """Stampa il report finale della validazione"""
        print("\n" + "=" * 80)
        print("REPORT VALIDAZIONE")
        print("=" * 80)

        if self.warnings:
            print(f"\n⚠️  WARNING ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.errors:
            print(f"\n❌ ERRORI CRITICI ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
            print("\n" + "=" * 80)
            print("⛔ VALIDAZIONE FALLITA - IL DATASET DEVE ESSERE NORMALIZZATO")
            print("=" * 80)
        else:
            print("\n✅ VALIDAZIONE COMPLETATA CON SUCCESSO")
            print("=" * 80)


def main():
    """Entry point per la validazione del dataset"""
    # Path del progetto
    project_root = Path(__file__).parent.parent
    knowledge_path = project_root / 'knowledge'
    training_path = project_root / 'training_data'

    # Crea e esegue il validator
    validator = DatasetValidator(
        knowledge_path=str(knowledge_path),
        training_path=str(training_path)
    )

    is_valid = validator.validate_all()

    # Exit con codice di errore se la validazione fallisce
    if not is_valid:
        print("\n💡 AZIONE RICHIESTA:")
        print("   1. Normalizza le entità NER in tutti i file YAML")
        print("   2. Rimuovi intenti duplicati")
        print("   3. Sposta esempi duplicati nell'intent corretto")
        print("   4. Esegui nuovamente la validazione")
        sys.exit(1)
    else:
        print("\n✅ Il dataset è pronto per il training!")
        sys.exit(0)


if __name__ == '__main__':
    main()

