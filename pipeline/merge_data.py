import json
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_intents_from_data(data: Any) -> List[Dict[str, Any]]:
    """Try several common formats and return a list of intent dicts.

    Expected intent dict: { 'intent': <name>, 'examples': [..] }
    """
    intents = []

    # case: {"nlu": {"intents": [ ... ]}}
    if isinstance(data, dict) and 'nlu' in data and isinstance(data['nlu'], dict) and 'intents' in data['nlu']:
        return data['nlu']['intents']

    # case: top-level {"intents": [ ... ]}
    if isinstance(data, dict) and 'intents' in data and isinstance(data['intents'], list):
        return data['intents']

    # case: list of intent objects: [ {"intent":..., "examples": [...]}, ... ]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and ('intent' in item or 'name' in item):
                intents.append(item)
        return intents

    # case: mapping intent_name -> [phrases]
    if isinstance(data, dict):
        # check if values are lists of strings
        mapping_like = True
        for k, v in data.items():
            if not isinstance(v, list):
                mapping_like = False
                break
        if mapping_like:
            for k, v in data.items():
                intents.append({'intent': k, 'examples': v})
            return intents

    return []


def _gather_examples(intent_entry: Dict[str, Any]) -> List[str]:
    # look for common keys
    for key in ('examples', 'utterances', 'phrases', 'examples_list'):
        if key in intent_entry and isinstance(intent_entry[key], list):
            return [str(x).strip() for x in intent_entry[key] if isinstance(x, str) and x.strip()]
    # sometimes examples are stored directly as a string
    if 'text' in intent_entry and isinstance(intent_entry['text'], str):
        return [intent_entry['text'].strip()]
    return []


def normalize_for_fasttext(text: str) -> str:
    """Normalize text for FastText: remove NER annotations and collapse spaces"""
    import re
    # Remove NER annotations like [Mario](PERSON) -> Mario
    text = re.sub(r'\[([^\]]+)\]\([A-Z_]+\)', r'\1', text)
    # Collapse spaces and strip, avoid tabs
    return ' '.join(text.split()).replace('\t', ' ')


def merge_intents(input_dir: str = "knowledge/intents", output_file: str = "data/training_source.json", fasttext_output: str = "data/fast-text.txt", fasttext_base: str = "knowledge/embeddings.txt", write_fasttext: bool = True) -> Dict[str, int]:
    """Merge intent files from `input_dir` into a single JSON file and optionally produce a FastText training file.

    The FastText output file is created by:
    1. First copying the base file (fasttext_base) if it exists
    2. Then appending examples from intent files

    This ensures that generic phrases in the base file are preserved.

    Returns a summary dict with counters for testing.
    """
    input_path = Path(input_dir)
    summary = {
        'files_total': 0,
        'files_ok': 0,
        'files_failed': 0,
        'intents_total': 0,
        'examples_total': 0,
        'examples_written': 0,
    }

    if not input_path.exists():
        logger.error(f"Directory {input_path} non trovata")
        return summary

    existing_intents = {}
    all_intents = []

    # Load existing output if present to preserve intent order
    out_path = Path(output_file)
    if out_path.exists():
        try:
            with out_path.open('r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if 'nlu' in existing_data and 'intents' in existing_data['nlu']:
                for intent in existing_data['nlu']['intents']:
                    existing_intents[intent['intent']] = intent
                all_intents = list(existing_intents.values())
        except Exception:
            logger.exception(f"Errore caricando esistente {output_file}, continuo con merge pulito")
            existing_intents = {}
            all_intents = []

    # Prepare fasttext output - merge with base file
    ft_path = Path(fasttext_output)
    ft_lines = []  # Collect all lines first
    if write_fasttext:
        ft_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy base fasttext file first (if exists)
        base_path = Path(fasttext_base)
        if base_path.exists():
            logger.info(f"Copiando frasi base da {fasttext_base} a {fasttext_output}")
            with base_path.open('r', encoding='utf-8') as base_f:
                for line in base_f:
                    line = line.strip()
                    if line:  # skip empty lines
                        ft_lines.append(line)
            logger.info(f"Frasi base copiate, ora aggiungo esempi dagli intenti...")
        else:
            logger.info(f"File base {fasttext_base} non trovato, creo file solo con esempi intenti")

    # diagnostics log
    diag_path = Path('diagnostics')
    diag_path.mkdir(exist_ok=True)
    diag_file = diag_path.joinpath('merge_errors.log')

    with diag_file.open('a', encoding='utf-8') as diag:
        for json_file in sorted(input_path.glob('*.json')):
            summary['files_total'] += 1
            try:
                logger.info(f"Caricando: {json_file}")
                with json_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)

                intents = _extract_intents_from_data(data)
                if not intents:
                    logger.warning(f"Nessun intent trovato in {json_file}")
                    summary['files_failed'] += 1
                    diag.write(f"{json_file}: no intents extracted\n")
                    continue

                for intent_entry in intents:
                    intent_name = intent_entry.get('intent') or intent_entry.get('name')
                    if not intent_name:
                        # skip malformed
                        continue
                    summary['intents_total'] += 1

                    examples = _gather_examples(intent_entry)
                    # fallback: some files use 'examples' inside an object named 'examples'
                    if not examples:
                        # try common alternative key names inside intent dict
                        for k in intent_entry.keys():
                            if isinstance(intent_entry[k], list) and all(isinstance(x, str) for x in intent_entry[k]):
                                examples = [str(x).strip() for x in intent_entry[k] if x.strip()]
                                break

                    if intent_name in existing_intents:
                        logger.debug(f"Intent '{intent_name}' già presente, salto aggiunta duplicata di intent (ma scrivo esempi in fasttext)")
                        intent_obj = existing_intents[intent_name]
                    else:
                        intent_obj = {'intent': intent_name, 'examples': []}
                        all_intents.append(intent_obj)
                        existing_intents[intent_name] = intent_obj

                    for ex in examples:
                        summary['examples_total'] += 1
                        cleaned = normalize_for_fasttext(ex)
                        if cleaned:
                            # append to merged json if not present
                            if 'examples' not in intent_obj:
                                intent_obj['examples'] = []
                            if ex not in intent_obj['examples']:
                                intent_obj['examples'].append(ex)
                            # write fasttext line (no labels - unsupervised mode)
                            if write_fasttext:
                                ft_lines.append(cleaned)
                                summary['examples_written'] += 1

                summary['files_ok'] += 1
            except Exception as e:
                logger.exception(f"Errore processando {json_file}")
                summary['files_failed'] += 1
                diag.write(f"{json_file}: exception {e}\n")

    # Write fasttext file without trailing newline
    if write_fasttext and ft_lines:
        with ft_path.open('w', encoding='utf-8') as ft_file:
            ft_file.write('\n'.join(ft_lines))

    # write merged JSON
    merged_data = {"nlu": {"intents": all_intents}}
    try:
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Dati mergiati salvati in: {output_file}")
        logger.info(f"Numero totale intenti: {len(all_intents)}")
    except Exception:
        logger.exception(f"Errore scrivendo {output_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge intent files from knowledge/intents")
    parser.add_argument("-i", "--input", default="knowledge/intents", help="Directory input con file JSON degli intent")
    parser.add_argument("-o", "--output", default="data/training_source.json", help="File JSON di output (default: data/training_source.json)")
    parser.add_argument("-f", "--fasttext", default="data/fast-text.txt", help="File di output per fastText (default: data/fast-text.txt)")
    parser.add_argument("-b", "--fasttext-base", default="knowledge/embeddings.txt", help="File base di frasi per fastText (default: knowledge/embeddings.txt)")
    parser.add_argument("--no-fasttext", action="store_true", help="Non generare il file per fastText")
    args = parser.parse_args()

    summary = merge_intents(args.input, args.output, args.fasttext, args.fasttext_base, write_fasttext=not args.no_fasttext)
    print(summary)
