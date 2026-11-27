"""
Esporta il modello NLU addestrato (Intent Classifier + NER) in un UNICO file ONNX, autosufficiente:
contiene il grafo della rete (decodifica CRF/Viterbi del NER compresa) e, nei metadati, tutto ciò che
serve per usarlo (tokenizer SentencePiece, mappa degli intenti, tag NER, soglia di confidenza).
Non servono né PyTorch né questo progetto.

Uso:
    python -m intellective.export_onnx                    # scrive models/cognitor-intent.onnx
    python -m intellective.export_onnx --output /percorso/cognitor-intent.onnx

Come usare il file (solo `onnxruntime` + `sentencepiece`):
    session = onnxruntime.InferenceSession(path)
    meta = session.get_modelmeta().custom_metadata_map
    labels = json.loads(meta["labels"])                       # lista ordinata delle classi (stesso ordine dei logits/probs)
    manifest = json.loads(meta["cognitor.manifest"])          # intents, labels, min_confidence, tokenizer, ...
    sp = SentencePieceProcessor(model_proto=base64.b64decode(meta["cognitor.tokenizer"]))
    ids = sp.encode(testo.lower(), out_type=int)              # lowercase: vedi manifest["tokenizer"]
    probs, tag_ids = session.run(["probs", "ner_tags"], {"input_ids": [ids]})
    # probs: float [1, n_intents] -> labels[probs[0].argmax()]; tag_ids: int64 [1, seq] -> manifest["ner_tags"][str(id)]

Le entità si ricostruiscono dai tag BIO sui token (pezzi SentencePiece senza il marcatore "▁"):
vedi README, sezione "Esportare il classificatore in ONNX".

Dopo l'export il file viene riaperto e confrontato con il modello PyTorch su alcune frasi (probabilità
degli intenti e tag NER), leggendo il tokenizer dal file stesso: se divergono l'export fallisce.

Funziona con il tokenizer SentencePiece (USE_FASTTEXT=false, default).
"""
import argparse
import base64
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.model_loader import ModelLoader
from classes.intent_factorizer import IntentFactorizer
from config import BASE_DIR, MIN_INTENT_CONFIDENCE, USE_FASTTEXT

MANIFEST_VERSION = 1
ONNX_FILENAME = "cognitor-intent.onnx"
LABELS_KEY = "labels"  # lista ordinata dei nomi degli intenti (stesso ordine degli indici di probs/logits)
NER_LABELS_KEY = "ner_labels"  # lista ordinata dei tag NER (stesso ordine degli ID di ner_tags)
TOPICS_KEY = "topics"  # solo modello a due teste: argomenti, nell'ordine di topic_probs
TYPES_KEY = "types"    # solo modello a due teste: tipi di entità, nell'ordine di type_probs ("" = nessun tipo)
MANIFEST_KEY = "cognitor.manifest"
TOKENIZER_KEY = "cognitor.tokenizer"  # sentencepiece.model in base64
INPUT_NAME = "input_ids"
OUTPUT_NAME = "probs"
NER_OUTPUT_NAME = "ner_tags"
TOPIC_OUTPUT_NAME = "topic_probs"  # solo modello a due teste: [batch, n_argomenti]
TYPE_OUTPUT_NAME = "type_probs"    # solo modello a due teste: [batch, n_tipi] (indice 0 = nessun tipo)
OPSET = 17

PARITY_SENTENCES = [
    "a che ora apre il centro unico di prenotazione?",
    "qual è il numero di telefono del pronto soccorso?",
    "come posso pagare il ticket sanitario?",
    "grazie mille",
    "ciao",
    "dove si trova l'ospedale più vicino a casa mia?",
    "vorrei prenotare una visita cardiologica al pronto soccorso",
    "come si rinnova la carta d'identità presso il comune?",
    "quali documenti servono per l'esenzione del ticket?",
    "a",
]
PARITY_TOLERANCE = 1e-4


@torch.jit.script
def viterbi_decode(emissions: torch.Tensor, start: torch.Tensor, end: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """
    Decodifica Viterbi di un CRF (stessi parametri di torchcrf.CRF), scritta in TorchScript per
    esportarla nel grafo ONNX come Loop. Sequenze senza padding. emissions [B, T, K] -> tag [B, T].
    """
    B, T, K = emissions.shape
    score = start.unsqueeze(0) + emissions[:, 0]
    history = torch.zeros(T, B, K, dtype=torch.long)
    for i in range(1, T):
        nxt = score.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, i].unsqueeze(1)
        score, idx = nxt.max(dim=1)
        history[i] = idx
    score = score + end.unsqueeze(0)
    best = score.argmax(dim=1)
    tags = torch.zeros(T, B, dtype=torch.long)
    tags[T - 1] = best
    for j in range(T - 1):
        i = T - 1 - j
        best = history[i].gather(1, best.unsqueeze(1)).squeeze(1)
        tags[i - 1] = best
    return tags.transpose(0, 1)


class NLUExport(nn.Module):
    """
    Intent (embedding -> BiGRU -> attention -> fc -> softmax) e NER (BiGRU -> fc -> Viterbi CRF)
    in un unico grafo. Restituisce (probs [B, n_intents], ner_tags [B, seq]); se il modello ha due teste
    (argomento e tipo di entità) restituisce anche (topic_probs [B, n_argomenti], type_probs [B, n_tipi]).
    `probs` resta la distribuzione sugli intenti completi (prodotto delle due teste), quindi chi non usa
    le due teste legge lo stesso output di prima.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        m = self.model
        gru_out, _ = m.bigru(m.embed_dropout(m.embedding(input_ids)))
        intent_logits, topic_logits, type_logits = m.intent_heads(m.dropout(m.attention(gru_out)))
        probs = F.softmax(intent_logits, dim=-1)
        emissions = m.ner_fc(m.ner_dropout(gru_out))
        tags = viterbi_decode(emissions, m.crf.start_transitions, m.crf.end_transitions, m.crf.transitions)
        if topic_logits is None:
            return probs, tags
        return probs, tags, F.softmax(topic_logits, dim=-1), F.softmax(type_logits, dim=-1)


def export(output_path: str) -> str:
    if USE_FASTTEXT:
        raise RuntimeError("L'export ONNX supporta solo il tokenizer SentencePiece (USE_FASTTEXT=false).")

    sp_model_path = os.path.join(BASE_DIR, ".cognitor", "sentencepiece.model")
    if not os.path.exists(sp_model_path):
        raise FileNotFoundError(f"{sp_model_path} non trovato: esegui prima il training (python -m pipeline)")

    loader = ModelLoader(BASE_DIR, torch.device("cpu"))
    intent_dict = loader.load_intent_dict()
    model, loaded = loader.load_intent_classifier(loader.load_vocab_size(), len(intent_dict))
    if not loaded:
        raise RuntimeError("Modello addestrato non trovato: esegui prima il training (python -m pipeline)")
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    factored = model.factored
    output_names = [OUTPUT_NAME, NER_OUTPUT_NAME] + ([TOPIC_OUTPUT_NAME, TYPE_OUTPUT_NAME] if factored else [])
    dynamic_axes = {
        INPUT_NAME: {0: "batch", 1: "seq"},
        OUTPUT_NAME: {0: "batch"},
        NER_OUTPUT_NAME: {0: "batch", 1: "seq"},
    }
    if factored:
        dynamic_axes.update({TOPIC_OUTPUT_NAME: {0: "batch"}, TYPE_OUTPUT_NAME: {0: "batch"}})

    dummy = torch.randint(1, 100, (2, 8), dtype=torch.long)
    torch.onnx.export(
        NLUExport(model).eval(),
        (dummy,),
        output_path,
        input_names=[INPUT_NAME],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
    )

    intents_by_id = {int(k): v for k, v in intent_dict.items()}
    labels = [intents_by_id[i] for i in range(len(intents_by_id))]
    tags_by_id = {int(v): k for k, v in model.ner_tag_builder.tag2id.items()}
    ner_labels = [tags_by_id[i] for i in range(len(tags_by_id))]

    manifest = {
        "version": MANIFEST_VERSION,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "ner_output_name": NER_OUTPUT_NAME,
        "tokenizer": {"type": "sentencepiece", "lowercase": True},
        "min_confidence": MIN_INTENT_CONFIDENCE,
        "temperature": round(float(model.temperature), 4),  # già applicata a `probs`: informativa
        "labels": labels,
        "ner_labels": ner_labels,
        "intents": {str(i): name for i, name in enumerate(labels)},
        "ner_tags": {str(i): tag for i, tag in enumerate(ner_labels)},
    }
    metadata_extra = {}
    if factored:
        factors = IntentFactorizer.load(loader.intent_factors_path)
        manifest.update({
            "factored": True,
            "topic_output_name": TOPIC_OUTPUT_NAME,
            "type_output_name": TYPE_OUTPUT_NAME,
            "topics": factors.topics,
            "types": factors.types,
            # per ogni intento (stesso ordine di labels): indice del suo argomento e del suo tipo
            "intent_topic": factors.pair_topic,
            "intent_type": factors.pair_type,
            # 1 se il tipo ha senso per l'intento (non per saluti, conferme...)
            "intent_type_applies": [int(a) for a in factors.type_applies],
        })
        metadata_extra = {TOPICS_KEY: json.dumps(factors.topics, ensure_ascii=False),
                          TYPES_KEY: json.dumps(factors.types, ensure_ascii=False)}
    with open(sp_model_path, "rb") as f:
        tokenizer_b64 = base64.b64encode(f.read()).decode("ascii")
    _embed_metadata(output_path, {
        LABELS_KEY: json.dumps(labels, ensure_ascii=False),
        NER_LABELS_KEY: json.dumps(ner_labels, ensure_ascii=False),
        **metadata_extra,
        MANIFEST_KEY: json.dumps(manifest, ensure_ascii=False),
        TOKENIZER_KEY: tokenizer_b64,
    })

    max_diff = verify_parity(model, output_path)
    print(f"OK Export completato: {output_path} "
          f"({os.path.getsize(output_path) / 1e6:.1f} MB, differenza massima vs PyTorch: {max_diff:.2e})")
    return output_path


def _embed_metadata(onnx_path: str, metadata: dict) -> None:
    """Scrive i metadati e porta i pesi dentro il file .onnx (nessun file di dati esterno)."""
    import onnx

    proto = onnx.load(onnx_path, load_external_data=True)
    del proto.metadata_props[:]
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save_model(proto, onnx_path, save_as_external_data=False)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)


def verify_parity(model, onnx_path: str) -> float:
    """
    Riapre il file esportato, ne legge tokenizer e manifest dai metadati e confronta con PyTorch su frasi
    di prova: probabilità degli intenti (entro PARITY_TOLERANCE) e tag NER (identici).
    """
    import numpy as np
    import onnxruntime as ort
    import sentencepiece as spm

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    labels = json.loads(meta[LABELS_KEY])
    ner_labels = json.loads(meta[NER_LABELS_KEY])
    manifest = json.loads(meta[MANIFEST_KEY])
    if labels != [manifest["intents"][str(i)] for i in range(len(labels))]:
        raise RuntimeError("Metadato 'labels' non allineato a manifest['intents']")
    if ner_labels != [manifest["ner_tags"][str(i)] for i in range(len(ner_labels))]:
        raise RuntimeError("Metadato 'ner_labels' non allineato a manifest['ner_tags']")
    sp = spm.SentencePieceProcessor(model_proto=base64.b64decode(meta[TOKENIZER_KEY]))

    max_diff = 0.0
    sentences_with_entities = 0
    for sentence in PARITY_SENTENCES:
        text = sentence.lower() if manifest["tokenizer"]["lowercase"] else sentence
        ids = sp.encode(text, out_type=int)
        expected = model.predict(sentence)
        factored = manifest.get("factored", False)
        outputs = session.run(
            [manifest["output_name"], manifest["ner_output_name"]]
            + ([manifest["topic_output_name"], manifest["type_output_name"]] if factored else []),
            {manifest["input_name"]: np.array([ids], dtype=np.int64)},
        )
        probs, tag_ids = outputs[0], outputs[1]

        diff = float(np.abs(np.array(expected["intent_probs"], dtype=np.float32) - probs[0]).max())
        max_diff = max(max_diff, diff)
        if diff > PARITY_TOLERANCE:
            raise RuntimeError(f"Export ONNX non allineato a PyTorch su '{sentence}' (intent, differenza {diff:.2e})")

        if factored:
            for name, got in (("topic_probs", outputs[2]), ("type_probs", outputs[3])):
                head_diff = float(np.abs(np.array(expected[name], dtype=np.float32) - got[0]).max())
                max_diff = max(max_diff, head_diff)
                if head_diff > PARITY_TOLERANCE:
                    raise RuntimeError(f"Export ONNX non allineato a PyTorch su '{sentence}' ({name}, differenza {head_diff:.2e})")

        tags = [manifest["ner_tags"][str(t)] for t in tag_ids[0]]
        if tags != expected["ner_tags"]:
            raise RuntimeError(f"Export ONNX non allineato a PyTorch su '{sentence}' (NER: {tags} != {expected['ner_tags']})")
        sentences_with_entities += any(t != "O" for t in tags)

    if not sentences_with_entities:
        raise RuntimeError("Le frasi di prova non contengono entità: il confronto NER non è significativo")
    return max_diff


def main():
    parser = argparse.ArgumentParser(description="Esporta l'Intent Classifier in ONNX")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "models", ONNX_FILENAME),
                        help=f"File di destinazione (default: models/{ONNX_FILENAME})")
    export(parser.parse_args().output)


if __name__ == "__main__":
    main()
