# [RISOLTO] NER: l'entità si interrompe a metà parola quando la parola è spezzata in più pezzi SentencePiece

**Stato: risolto** con la soluzione 1 (`IntentClassifier._snap_to_word_boundaries`, usata da `_extract_entities`).
Non serve riaddestrare né riesportare: il grafo ONNX non cambia, cambia solo la ricostruzione delle entità dai tag
(da applicare anche in chi usa il file ONNX, vedi README). Test: `tests/test_ner_word_boundaries.py`.

## Problema

Con il modello addestrato, il NER restituisce entità troncate nel mezzo di una parola.

Esempio:

```
testo:  come si rinnova la carta d'identità presso il comune?
atteso: DOCUMENT  "carta d'identità"   (start 19, end 35)
ottenuto: DOCUMENT "carta d'i"          (start 19, end 28)
```

Tag per token (output di `model.predict`):

```
carta     B-DOCUMENT
d         I-DOCUMENT
'         I-DOCUMENT
i         I-DOCUMENT
dentità   O              <-- "identità" è spezzata in "i" + "dentità" e il secondo pezzo è O
```

Con la forma senza apostrofo, `come rinnovo carta d identità elettronica scadenza`, "identità" viene spezzata
nello stesso modo (`i` + `dentità`) ma qui entrambi i pezzi sono `I-DOCUMENT` e l'entità esce corretta
(`carta d identità elettronica`). Quindi il problema è incostante e dipende dai tag che il modello assegna ai singoli pezzi.

## Come riprodurlo

```python
from agent.model_loader import ModelLoader
import torch

loader = ModelLoader(".", torch.device("cpu"))
intent_dict = loader.load_intent_dict()
model, _ = loader.load_intent_classifier(loader.load_vocab_size(), len(intent_dict))

s = "come si rinnova la carta d'identità presso il comune?"
r = model.predict(s)
print(list(zip(r["tokens"], r["ner_tags"])))
print(model._extract_entities(r["tokens"], r["ner_tags"], s))
```

Oppure `python scripts/test_nlu.py "come si rinnova la carta d'identità presso il comune?"`.

## Note

- Nel dataset (`.cognitor/nlu_data.csv`) ci sono 24 esempi con "d'identità"; per esempio
  `come si rinnova la carta d'identità` ha l'entità annotata `"carta d'identità"` (offset 19–35), quindi
  l'annotazione di training è corretta.
- Il difetto è presente anche nel modello esportato in ONNX (`intellective/export_onnx.py`): su 409 frasi
  l'output ONNX è identico a quello di PyTorch, quindi non è un problema dell'export.

## Ipotesi (non verificate)

- Il modello assegna `O` al secondo pezzo di una parola spezzata. Le entità possono quindi finire dentro una parola.
- L'allineamento dei tag BIO ai pezzi in training (`NERTagBuilder.align_tokens_to_bio`) potrebbe trattare in modo
  diverso i pezzi di continuazione della parola, rispetto a quelli che iniziano una parola.

## Possibili soluzioni

1. In `_extract_entities`, estendere la fine di ogni entità fino al confine di parola nel testo originale
   (fino al prossimo spazio o segno di punteggiatura). È una correzione a valle, semplice, che non richiede nuovo training.
2. Propagare il tag di un pezzo di inizio parola a tutti i pezzi di continuazione della stessa parola, sia in training
   sia in inferenza.
3. Aggiungere esempi di training con parole spezzate in più pezzi (apostrofi come `d'identità`, `dell'esenzione`).

## Verifica quando è risolto

Un test che controlli che, per ogni entità restituita, `start` e `end` cadano su confini di parola del testo originale
(carattere precedente a `start` e successivo a `end` non alfanumerici), eseguito sulle frasi di `nlu_data.csv` e sulla
frase dell'esempio qui sopra.
