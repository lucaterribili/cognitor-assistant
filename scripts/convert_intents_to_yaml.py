"""
Script per convertire tutti i file intents da JSON a YAML.
Converte automaticamente il formato mantenendo la struttura.
"""
import json
import yaml
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INTENTS_DIR = BASE_DIR / 'knowledge' / 'intents'


def convert_intents_json_to_yaml(json_file: Path) -> None:
    """Converte un file intents.json in formato YAML."""

    # Leggi JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Crea YAML
    yaml_file = json_file.with_suffix('.yaml')

    # Scrivi YAML con configurazione pulita
    with open(yaml_file, 'w', encoding='utf-8') as f:
        # Header
        yaml_name = json_file.stem.replace('_', ' ').title()
        f.write(f"# {yaml_name} Intents - Training examples\n\n")

        # Scrivi la struttura
        yaml.dump(data, f,
                  default_flow_style=False,
                  allow_unicode=True,
                  sort_keys=False,
                  width=1000)  # Evita line wrapping

    print(f"✓ Convertito: {json_file.name} → {yaml_file.name}")


def main():
    print("="*60)
    print("CONVERSIONE INTENTS: JSON → YAML")
    print("="*60)

    # Trova tutti i file JSON nella cartella intents
    json_files = list(INTENTS_DIR.glob('*.json'))

    if not json_files:
        print("\n✗ Nessun file JSON trovato in knowledge/intents/")
        return

    print(f"\nTrovati {len(json_files)} file JSON da convertire:\n")

    for json_file in json_files:
        convert_intents_json_to_yaml(json_file)

    print("\n" + "="*60)
    print("✅ CONVERSIONE COMPLETATA!")
    print("="*60)
    print(f"\nConvertiti {len(json_files)} file da JSON a YAML")
    print("\nI file JSON originali sono stati mantenuti.")
    print("Il sistema caricherà automaticamente i file YAML.")


if __name__ == "__main__":
    main()

