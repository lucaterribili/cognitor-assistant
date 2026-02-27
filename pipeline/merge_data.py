import json
import os
import argparse
from pathlib import Path


def merge_training_data(output_file: str = "data.json"):
    """
    Merge di tutti i file JSON nella cartella training_data/ in un unico data.json
    Mantiene gli intents esistenti in data.json e aggiunge quelli nuovi da base.json
    """
    training_dir = Path("training_data")
    
    if not training_dir.exists():
        print(f"Directory {training_dir} non trovata")
        return

    existing_intents = {}
    
    if Path(output_file).exists():
        print(f"Caricando intents esistenti da: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        if 'nlu' in existing_data and 'intents' in existing_data['nlu']:
            for intent in existing_data['nlu']['intents']:
                existing_intents[intent['intent']] = intent
    
    all_intents = list(existing_intents.values())
    
    for json_file in training_dir.glob("*.json"):
        if json_file.name == output_file:
            continue
            
        print(f"Caricando: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'nlu' in data and 'intents' in data['nlu']:
            for intent in data['nlu']['intents']:
                if intent['intent'] not in existing_intents:
                    all_intents.append(intent)
                    existing_intents[intent['intent']] = intent
                else:
                    print(f"  - Intent '{intent['intent']}' già presente, salto")
    
    merged_data = {"nlu": {"intents": all_intents}}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDati mergiati salvati in: {output_file}")
    print(f"Numero totale intenti: {len(all_intents)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge training data files")
    parser.add_argument("-o", "--output", default="data.json", help="Output file (default: data.json)")
    args = parser.parse_args()
    
    merge_training_data(args.output)
