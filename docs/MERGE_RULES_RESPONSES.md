# Merge di Rules e Responses nella cartella `.cognitor`

## Panoramica

Implementato il sistema di merge per rules e responses analogamente a quanto già fatto per gli intenti. Ora tutta la conoscenza del bot (intenti, rules, responses) viene consolidata nella cartella `.cognitor` prima del training e dell'esecuzione.

## Struttura Directory

### Directory Sorgente
La conoscenza del bot è distribuita in due cartelle principali:

```
knowledge/
  ├── intents/      # Intenti base del sistema
  ├── rules/        # Regole DSL base
  └── responses/    # Risposte template base

training_data/
  ├── intents/      # Intenti personalizzati/di test
  ├── rules/        # Regole DSL personalizzate
  └── responses/    # Risposte template personalizzate
```

### Directory Target (`.cognitor`)
Dopo il merge, tutti i file vengono consolidati in:

```
.cognitor/
  ├── rules.yaml           # Tutte le rules mergiate
  ├── responses.yaml       # Tutte le responses mergiate
  ├── intent_dict.json     # Dizionario intenti (generato dalla pipeline)
  ├── nlu_data.csv         # Dataset NLU tokenizzato
  ├── fast-text.txt        # Corpus per FastText
  └── ...
```

## Funzionalità Implementate

### 1. Funzione `merge_rules()`

**File**: `pipeline/merge_data.py`

```python
def merge_rules(input_dirs: List[str] = None, output_file: str = ".cognitor/rules.yaml") -> Dict[str, int]
```

**Comportamento**:
- Legge tutti i file `.yaml` e `.yml` dalle directory specificate
- Default: `["knowledge/rules", "training_data/rules"]`
- Merge di tutte le rules in un unico file YAML
- In caso di conflitti, l'ultimo file letto sovrascrive il precedente (con warning)
- Output: `.cognitor/rules.yaml`

### 2. Funzione `merge_responses()`

**File**: `pipeline/merge_data.py`

```python
def merge_responses(input_dirs: List[str] = None, output_file: str = ".cognitor/responses.yaml") -> Dict[str, int]
```

**Comportamento**:
- Legge tutti i file `.yaml` e `.yml` dalle directory specificate
- Default: `["knowledge/responses", "training_data/responses"]`
- Merge di tutte le responses in un unico file YAML
- In caso di conflitti, l'ultimo file letto sovrascrive il precedente (con warning)
- Output: `.cognitor/responses.yaml`

### 3. Aggiornamento `KnowledgeLoader`

**File**: `agent/model_loader.py`

Il `KnowledgeLoader` è stato aggiornato per:
- Leggere i file mergiati da `.cognitor/rules.yaml` e `.cognitor/responses.yaml`
- Mantenere compatibilità con la modalità legacy (lettura diretta da `knowledge/`)
- Fallback automatico se i file mergiati non esistono

### 4. Integrazione nella Pipeline

**File**: `pipeline/__init__.py`

La funzione `run_full_pipeline()` ora include lo step 1.5:

```
[1.5/5] Merge rules e responses in .cognitor...
  ✓ Rules mergiati: N da X file(s)
  ✓ Responses mergiati: N da Y file(s)
```

Questo step viene eseguito dopo la generazione del corpus FastText e prima del training.

## Utilizzo

### Da CLI

```bash
# Merge solo rules
python pipeline/merge_data.py --type rules

# Merge solo responses
python pipeline/merge_data.py --type responses

# Merge tutto (intents, rules, responses)
python pipeline/merge_data.py --type all
```

### Da Codice Python

```python
from pipeline import merge_rules, merge_responses

# Merge con directory di default
rules_summary = merge_rules()
responses_summary = merge_responses()

# Merge con directory personalizzate
rules_summary = merge_rules(
    input_dirs=["custom/rules", "other/rules"],
    output_file=".cognitor/rules.yaml"
)
```

### Pipeline Completa

```bash
# Esegue tutta la pipeline incluso il merge
python -m pipeline
```

## Vantaggi

1. **Centralizzazione**: Tutta la conoscenza del bot in un unico posto (`.cognitor`)
2. **Separazione**: Distingue tra conoscenza base (`knowledge/`) e personalizzazioni (`training_data/`)
3. **Consistenza**: Stesso approccio per intenti, rules e responses
4. **Performance**: Lettura più veloce (un solo file invece di N file)
5. **Debugging**: Più facile vedere tutta la configurazione in un colpo d'occhio
6. **Versionamento**: Il contenuto di `.cognitor` può essere escluso dal git (è generato)

## Esempio di Workflow

1. **Sviluppo**: Aggiungi nuove rules/responses in `training_data/`
   ```yaml
   # training_data/rules/my_feature.yaml
   rules:
     my_new_intent:
       default: my_new_response
   ```

2. **Merge**: Esegui il merge manualmente o tramite pipeline
   ```bash
   python pipeline/merge_data.py --type all
   ```

3. **Test**: Il sistema usa automaticamente i file mergiati da `.cognitor/`

4. **Deploy**: Una volta stabilizzate, sposta le regole in `knowledge/` per farle diventare parte del core

## Note Tecniche

- I file YAML preservano l'ordine di inserimento
- Il formato YAML è human-readable e facilita il debug
- Supporto per `allow_unicode=True` per caratteri speciali
- Logging dettagliato per tracciare quali file vengono processati
- Gestione robusta degli errori con diagnostica

## Compatibilità

- ✅ Compatibile con versioni precedenti (fallback su `knowledge/`)
- ✅ Supporta sia `.yaml` che `.yml`
- ✅ Integrato nella pipeline esistente
- ✅ Funziona con il sistema DSL attuale

