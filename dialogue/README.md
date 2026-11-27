# Conversazioni: corpus e modello dei flussi

Il comportamento del bot nella conversazione non è scritto nel codice: sta in un **corpus di conversazioni**
(`corpus/*.yaml`). Un modello impara da quel corpus cosa fa l'utente e cosa fa il bot, e genera conversazioni nuove
per un sito. **Per cambiare il comportamento si cambia il corpus.**

Questo file spiega come leggere, scrivere, controllare ed estendere il corpus.

## Comandi

Da eseguire nella cartella principale di Cognitor, con il suo ambiente virtuale attivo.

| Comando | A cosa serve |
|---|---|
| `python -m dialogue corpus-check` | controlla che i file siano corretti; dà un codice di uscita diverso da 0 se ci sono errori |
| `python -m dialogue corpus-stats` | conteggi per mossa, azione, argomento e transizione: mostra dove servono altri esempi |
| `python -m dialogue sample --n 10 --seed 1` | stampa conversazioni nuove generate dal modello, da leggere e correggere |
| `python -m dialogue export-model --out <file>.json` | scrive il modello (tabelle di conteggi) per l'engine, che lo usa per generare le conversazioni di un sito |
| `pytest tests/test_dialogue_corpus.py` | test del corpus e del modello |

## Il formato

Ogni file in `corpus/` ha questa forma:

```yaml
conversations:
  - id: slot-01                # unico in tutto il corpus
    source: llm                # hand = scritta a mano, llm = scritta da un LLM
    reviewed: false            # metti true dopo averla letta e approvata
    turns:
      - {u: "A che ora apre?", move: question, topic: opening_hours, bot: ask_slot}
      - {u: "Il pronto soccorso", move: slot_reply, ent: [DEPARTMENT], bot: answer}
      - {u: "Grazie", move: thanks, bot: thanks_reply}
```

Un turno è una riga. Il bot risponde all'utente: `bot` è l'**azione giusta** a quel messaggio.

| Campo | Obbligatorio | Significato |
|---|---|---|
| `u` | sì | cosa scrive l'utente (testo, serve a chi legge) |
| `move` | sì | che cosa fa l'utente (vedi sotto) |
| `bot` | sì | che cosa fa il bot (vedi sotto) |
| `topic` | per `question` | l'argomento della tassonomia, senza `ask_` (es. `opening_hours`, `phone`) |
| `ent` | no | tipi di entità **nominati nel messaggio** (es. `[DEPARTMENT]`) |
| `carry` | no | tipi di entità **ripresi dal contesto**, non nominati ora (es. "E il telefono?") |
| `rag` | no | esito del retrieval: `ok` (default), `low` (incerto), `none` (nessun contenuto) |

Gli argomenti e i tipi di entità devono esistere: gli argomenti sono quelli in `docs/INTENT_TAXONOMY.md` (nome del file
in `knowledge/intents/` senza `.yaml`), i tipi sono quelli del NER (`DEPARTMENT`, `FACILITY`, `DOCUMENT`...).

### Mosse dell'utente (`move`)

| Mossa | Quando |
|---|---|
| `question` | domanda su un argomento, con o senza entità |
| `slot_reply` | risponde a "di quale...?": dice solo l'entità (serve `ent`, dopo un `ask_slot`) |
| `follow_up_entity` | "E il distretto 2?": solo una nuova entità, l'argomento è quello di prima (serve `ent`, dopo una risposta) |
| `greeting`, `thanks`, `goodbye` | saluto, ringraziamento, congedo |
| `confirm`, `deny` | "sì / ok" e "no / non è quello" |
| `insult` | insulto |
| `handoff` | vuole un operatore |
| `repeat_request` | "non ho capito, puoi ripetere?" |
| `unclear` | messaggio incomprensibile |
| `out_of_scope` | fuori argomento |

### Azioni del bot (`bot`)

| Azione | Cosa fa il bot |
|---|---|
| `answer` | risponde alla domanda (regola, RAG o modello QA) |
| `ask_slot` | chiede l'entità che manca ("di quale dipartimento parli?") |
| `clarify` | chiede di riformulare o di precisare |
| `fallback` | ammette di non sapere (fuori argomento o nessun contenuto) |
| `greet` / `goodbye` | saluta / chiude |
| `thanks_reply` | risponde a un ringraziamento ("prego, serve altro?") |
| `offer_more` | "posso aiutarti con altro?" |
| `prompt` | "dimmi pure" |
| `repeat` | ripete l'ultima risposta |
| `handoff` | passa a un operatore |
| `deescalate` | risponde a un insulto |

## Come estendere il corpus

1. **Guarda dove manca.** `python -m dialogue corpus-stats`: azioni, mosse e transizioni con pochi esempi (la lista
   in coda è quella delle transizioni più rare) sono quelle da coprire.
2. **Leggi cosa genera il modello.** `python -m dialogue sample --n 10`: se un flusso ti sembra sbagliato, di solito
   manca un esempio giusto, o ce n'è uno sbagliato.
3. **Scrivi o correggi.** Aggiungi conversazioni a un file esistente o creane uno nuovo in `corpus/` (ogni `.yaml` viene
   letto). Una conversazione per situazione; 3-8 turni; testo realistico.
4. **Controlla.** `python -m dialogue corpus-check`: deve dire `0 errori`.
5. **Segna le revisioni.** Quando hai letto e approvato una conversazione, `reviewed: true`. Quelle scritte da te:
   `source: hand`.
6. **Rilancia i test.** `pytest tests/test_dialogue_corpus.py`.

Consigli:
- Più esempi della stessa situazione con testi diversi, non lo stesso testo copiato.
- Mostra anche i casi in cui il bot **non** risponde: manca l'entità, retrieval incerto (`rag: low`), nessun contenuto
  (`rag: none`), messaggi incomprensibili. Sono quelli che insegnano al modello quando fermarsi.
- L'azione giusta è quella che vuoi che il bot faccia, non quella che farebbe oggi.

## Errori comuni del controllo

| Messaggio | Causa |
|---|---|
| `argomento 'x' non presente nella tassonomia` | l'argomento non esiste: controlla il nome in `docs/INTENT_TAXONOMY.md` |
| `tipo di entità 'X' sconosciuto` | il tipo non è tra quelli del NER |
| `azione 'x' sconosciuta` / `mossa 'x' sconosciuta` | refuso nel nome |
| `slot_reply senza un ask_slot ancora senza risposta o senza entità` | una `slot_reply` deve seguire un turno con `bot: ask_slot` e avere `ent` |
| `follow_up_entity senza una risposta precedente o senza entità` | serve un `answer` prima, e `ent` |
| `carry 'X' ma nessun turno precedente lo nomina` | si può riprendere dal contesto solo un'entità già nominata con `ent` |
| `la mossa 'x' non ha un argomento` | `topic` si usa solo con `question` |
| `id duplicato` | ogni `id` deve essere unico in tutto il corpus |

## Come funziona il modello

`flow_model.py` stima dal corpus, con conteggi:
- quale mossa fa l'utente dopo una certa azione del bot e in una certa situazione (c'è uno slot in sospeso? c'è già stata
  una risposta?);
- quale azione fa il bot a quella mossa;
- quali argomenti compaiono con quali azioni.

Poi campiona una conversazione astratta (mosse e azioni) e `instantiate` la riempie con argomenti e valori di entità
del sito. Una mossa è ammessa solo se ha senso (una risposta a uno slot presuppone una domanda sullo slot, e così via):
sono le stesse condizioni del controllo.

Non è una rete neurale perché con poche decine di conversazioni una rete memorizzerebbe il corpus. L'interfaccia
(`fit` / `sample`) è pensata per sostituirlo con una GRU quando i dati lo giustificano.

## Altri file

| File | Cosa contiene |
|---|---|
| `actions.py` | elenco di azioni e mosse |
| `corpus.py` | caricamento, controllo e statistiche del corpus |
| `flow_model.py` | il modello dei flussi e `instantiate` |
| `taxonomy.py` | argomenti e tipi di entità, dal modello di Cognitor (`.cognitor/intent_factors.json`) |
| `dev_set.yaml` | conversazioni di sviluppo scritte a mano, indipendenti dal generatore |
| `simulator.py`, `rules.py`, `topic_config.yaml` | simulatore a regole: serve solo come baseline di confronto, non è la fonte di verità e si può eliminare |

## Cosa non c'è ancora

- L'uso nell'engine: oggi il modello si esegue solo da qui (`sample`). Mancano l'esportazione del modello, il passaggio
  dei valori veri del sito e la conversione nel formato di Cognitor (`conversations.yaml`).
- Una rete neurale al posto del modello a conteggi.
