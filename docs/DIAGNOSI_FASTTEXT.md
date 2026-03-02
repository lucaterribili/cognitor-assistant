# 🔬 DIAGNOSI: Problema Sequenze Corte

## ❌ PROBLEMA CONFERMATO

**FastText NON discrimina bene le parole corte!**

### Evidenze:

1. **Similarità troppo alta tra parole diverse:**
   - `ciao` ↔ `numero`: 0.66 (dovrebbe essere <0.3)
   - `ok` ↔ `capitale`: 0.72 (dovrebbe essere <0.3) 
   - `stop` ↔ `meteo`: 0.59 (dovrebbe essere <0.3)

2. **Causa principale:**
   - Corpus di training troppo piccolo: **2753 linee**
   - FastText necessita di almeno 100k-1M frasi per embeddings di qualità
   - Il vocabolario è di sole 2603 parole

3. **Training set:**
   - 65% sono sequenze ≤3 token (quindi il modello HA visto esempi corti)
   - Il problema NON è la mancanza di dati corti
   - Il problema È la qualità degli embeddings FastText

---

## 💡 SOLUZIONI

### ⭐ SOLUZIONE 1: Embeddings Pre-trainati (RACCOMANDATA)

**La più veloce ed efficace!**

Scarica embeddings italiani già trainati su grandi corpora:

**Opzione A: word2vec-google-news-300 (italiano)**
```bash
# Usa gensim per scaricare
pip install gensim
```

**Opzione B: FastText pre-trainati Facebook**
```bash
# Download dal sito Facebook Research
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz
gunzip cc.it.300.bin.gz
```

**Implementazione:**
1. Sostituisci il caricamento in `intent_classifier.py`
2. Usa il modello pre-trainato invece del tuo
3. Ri-traila l'intent classifier (senza ri-trainare gli embeddings)

---

### 🔧 SOLUZIONE 2: Espandi il Corpus FastText

**Richiede più tempo ma mantiene il controllo**

1. **Scarica corpus italiani:**
   - Wikipedia italiana (500k+ articoli)
   - OSCAR corpus (Common Crawl)
   - OpenSubtitles italiano

2. **Aggiungi al corpus:**
   ```bash
   cat data/fast-text.txt wiki_italiano.txt > data/fast-text-large.txt
   ```

3. **Ri-traila FastText:**
   ```python
   # Modifica train_fast_text.py:
   model = fasttext.train_unsupervised(
       input='data/fast-text-large.txt',
       model='skipgram',
       dim=300,
       epoch=50,  # Aumenta epoche
       lr=0.05,   # Diminuisci learning rate
       minCount=5, # Aumenta threshold
       wordNgrams=3,
       minn=2,
       maxn=5,
       ws=5
   )
   ```

---

### 🔨 SOLUZIONE 3: Hybrid Architecture (VELOCE)

**Aggiungi un branch per sequenze corte che bypassa FastText problematico**

Modifica `IntentClassifier` per gestire diversamente le sequenze da 1-2 token:

```python
# In forward():
if seq_len <= 2:
    # Usa embeddings trainabili separati per parole comuni corte
    x_direct = self.short_word_embeddings(x)  # Nuovo layer
    intent_logits = self.short_fc(x_direct.mean(dim=1))
else:
    # Usa il path normale con FastText + GRU
    ...
```

Questa soluzione aggira il problema senza ri-trainare FastText.

---

### 🎯 SOLUZIONE 4: Lookup Table per Comandi Base

**Più semplice per comandi fissi**

Crea un dizionario diretto per le ~20 parole corte più comuni:

```python
SHORT_WORD_INTENTS = {
    'ciao': 'greetings',
    'ok': 'confirm', 
    'stop': 'stop',
    'aiuto': 'help',
    'sì': 'confirm',
    'no': 'deny',
    # ...
}

# In predict():
if len(tokens) == 1 and tokens[0] in SHORT_WORD_INTENTS:
    return SHORT_WORD_INTENTS[tokens[0]]
```

---

## 🏆 RACCOMANDAZIONE

**Ordine di priorità:**

1. **SOLUZIONE 1** (embeddings pre-trainati) - Migliore qualità, setup veloce
2. **SOLUZIONE 4** (lookup table) - Velocissima per casi comuni
3. **SOLUZIONE 3** (hybrid architecture) - Buon compromesso
4. **SOLUZIONE 2** (espandi corpus) - Richiede molto tempo

---

## 📊 Dati Tecnici

```
Corpus attuale:
- Linee: 2,753
- Vocabolario: 2,603 parole
- Troppo piccolo per FastText di qualità

Training set:
- Esempi totali: 2,315
- Sequenze ≤3 token: 1,513 (65%)
- Il modello HA visto sequenze corte

FastText quality:
- Parole diverse hanno sim > 0.5 ❌
- Embeddings troppo simili tra loro ❌
- Necessita corpus 100x più grande ❌
```

---

## 🚀 Prossimi Passi

Vuoi che implementi:
- **A** - Scarico e integro embeddings pre-trainati? 
- **B** - Aggiungo hybrid architecture per seq corte?
- **C** - Creo lookup table per parole comuni?
- **D** - Ti aiuto a espandere il corpus?

