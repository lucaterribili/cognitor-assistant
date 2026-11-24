"""Valida knowledge/intents/*.yaml: formato, entita', duplicati, intenti poveri di esempi."""
import collections
import re
import sys
from pathlib import Path

import yaml

INTENTS_DIR = Path(__file__).parent.parent / 'knowledge' / 'intents'
MIN_EXAMPLES = 15
# Label delle entita' (tutte in inglese). Possono essere specifiche del dominio (il flusso e' gestito
# dal RAG), ma lo stesso testo ha sempre la stessa label e si etichettano solo entita' specifiche:
# i nomi generici (documenti, servizio, medico, ufficio...) NON si etichettano.
ENTITIES = {
    # generiche
    'PERSON', 'ROLE', 'ORGANIZATION', 'DATE', 'TIME', 'NUMBER', 'DOCUMENT', 'PRODUCT', 'EVENT',
    'PROCEDURE', 'LANGUAGE', 'CATEGORY', 'DISH', 'TAX', 'MISC',
    # luoghi: LOCATION solo se nessuna delle specifiche si applica
    'LOCATION', 'ADDRESS', 'COUNTRY', 'ZONE',
    # ASL (sanita')
    'FACILITY', 'DEPARTMENT', 'HEALTH_SERVICE', 'MEDICAL_PROCEDURE', 'TRANSPORT', 'PAYMENT_METHOD',
    # programmato (tech)
    'TECHNOLOGY', 'TECH_CONCEPT',
}
GENERIC_NOUNS = {
    'documento', 'documenti', 'documentazione', 'modulo', 'moduli', 'modulistica', 'domanda', 'richiesta',
    'file', 'pdf', 'link', 'guida', 'manuale', 'certificato', 'servizio', 'servizi', 'prestazione',
    'prestazioni', 'visita', 'visite', 'esame', 'esami', 'analisi', 'medico', 'medici', 'paziente',
    'pazienti', 'cittadini', 'personale', 'responsabile', 'ufficio', 'uffici', 'struttura', 'centro',
    'sede', 'ospedale', 'farmacia', 'azienda', 'ente', 'prodotto', 'prodotti', 'evento', 'eventi',
    'pratica', 'procedura',
}
ART = re.compile(r"^(il|lo|la|l'|i|gli|le|un|uno|una|un'|questo|questa|questi|queste)\b", re.I)
TAG = re.compile(r'\[([^\]]+)\]\(([A-Z_]+)\)')


def plain(e):
    return TAG.sub(lambda m: m.group(1), e)


def write_doc():
    """Rigenera docs/INTENT_TAXONOMY.md da knowledge/intents/*.yaml."""
    rows = ['| Intento | File | Esempi | Entita |', '|---|---|---|---|']
    total = 0
    for f in sorted(INTENTS_DIR.glob('*.yaml')):
        for it in yaml.safe_load(f.read_text(encoding='utf-8'))['nlu']['intents']:
            ents = collections.Counter(m.group(2) for e in it['examples'] for m in TAG.finditer(e))
            rows.append(f"| `{it['intent']}` | `{f.name}` | {len(it['examples'])} | {', '.join(k for k, _ in ents.most_common(4))} |")
            total += len(it['examples'])
    head = ['# Intenti base\n',
            f'{len(rows) - 2} intenti, {total} esempi. Sorgente di verita\': `knowledge/intents/*.yaml`.\n',
            "Gli intenti sono organizzati per file in base al suffisso e splittati per entita' dominante. "
            'Rigenera con `python scripts/validate_intents.py --doc`.\n']
    (Path(__file__).parent.parent / 'docs' / 'INTENT_TAXONOMY.md').write_text('\n'.join(head + rows) + '\n', encoding='utf-8')


def main():
    if '--doc' in sys.argv:
        write_doc()
    errors, owner, counts = [], {}, {}
    surf = collections.defaultdict(lambda: collections.defaultdict(set))
    for f in sorted(INTENTS_DIR.glob('*.yaml')):
        for it in yaml.safe_load(f.read_text(encoding='utf-8'))['nlu']['intents']:
            name, exs = it['intent'], it['examples']
            counts[name] = len(exs)
            for e in exs:
                if not isinstance(e, str) or not e.strip():
                    errors.append(f'{name}: esempio vuoto/non stringa {e!r}')
                    continue
                for m in TAG.finditer(e):
                    if m.group(2) not in ENTITIES:
                        errors.append(f'{name}: entita sconosciuta {m.group(2)} in {e!r}')
                    if m.group(1).lower().strip() in GENERIC_NOUNS:
                        errors.append(f'{name}: nome generico etichettato {m.group(1)!r} in {e!r}')
                    if ART.match(m.group(1)):
                        errors.append(f'{name}: span che inizia con un articolo {e!r}')
                    surf[m.group(1).lower().strip()][m.group(2)].add(name)
                    if m.group(1) != m.group(1).strip():
                        errors.append(f'{name}: spazi nell\'entita {e!r}')
                if re.search(r'[\[\]]', plain(e)):
                    errors.append(f'{name}: parentesi quadre non valide {e!r}')
                k = plain(e).lower().strip()
                if k in owner:
                    kind = 'duplicato' if owner[k] == name else f'AMBIGUO con {owner[k]}'
                    errors.append(f'{name}: {kind}: {e!r}')
                owner.setdefault(k, name)
    for text, types in surf.items():
        if len(types) > 1:
            errors.append(f'stesso testo, tipi diversi: {text!r} -> {sorted(types)}')
    low = {n: c for n, c in counts.items() if c < MIN_EXAMPLES}
    print(f'{len(counts)} intenti, {sum(counts.values())} esempi')
    if low:
        print(f'sotto {MIN_EXAMPLES} esempi: ' + ', '.join(f'{n}({c})' for n, c in sorted(low.items(), key=lambda x: x[1])))
    for e in errors:
        print('ERRORE', e)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
