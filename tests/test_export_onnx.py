import base64
import json
import os

import numpy as np
import onnxruntime as ort
import pytest
import sentencepiece as spm

from config import BASE_DIR
from intellective import export_onnx

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(BASE_DIR, "models", "intent_model_fast.pth")),
    reason="modello addestrato non presente: esegui prima python -m pipeline",
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / export_onnx.ONNX_FILENAME
    export_onnx.export(str(path))  # fallisce da solo se l'ONNX diverge da PyTorch
    return path


def test_export_is_a_single_self_contained_file(exported):
    assert [p.name for p in exported.parent.iterdir()] == [export_onnx.ONNX_FILENAME]

    meta = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    labels = json.loads(meta[export_onnx.LABELS_KEY])
    ner_labels = json.loads(meta[export_onnx.NER_LABELS_KEY])
    manifest = json.loads(meta[export_onnx.MANIFEST_KEY])
    assert manifest["tokenizer"]["type"] == "sentencepiece"
    assert isinstance(labels, list) and len(labels) == len(manifest["intents"])
    assert labels == [manifest["intents"][str(i)] for i in range(len(labels))]
    assert manifest["labels"] == labels
    assert isinstance(ner_labels, list) and ner_labels[0] == "O"
    assert ner_labels == [manifest["ner_tags"][str(i)] for i in range(len(ner_labels))]
    assert manifest["intents"]["0"] == labels[0]
    assert manifest["ner_tags"]["0"] == "O"
    sp = spm.SentencePieceProcessor(model_proto=base64.b64decode(meta[export_onnx.TOKENIZER_KEY]))
    assert sp.encode("ciao", out_type=int)


def test_exported_model_accepts_variable_batch_and_length(exported):
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    manifest = json.loads(session.get_modelmeta().custom_metadata_map[export_onnx.MANIFEST_KEY])
    n_intents, n_tags = len(manifest["intents"]), len(manifest["ner_tags"])
    for batch, seq in [(1, 1), (1, 12), (3, 7)]:
        ids = np.random.randint(1, 100, (batch, seq)).astype(np.int64)
        probs, tags = session.run([export_onnx.OUTPUT_NAME, export_onnx.NER_OUTPUT_NAME], {export_onnx.INPUT_NAME: ids})
        assert probs.shape == (batch, n_intents)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert tags.shape == (batch, seq)
        assert tags.min() >= 0 and tags.max() < n_tags


def test_two_head_export_has_topic_and_type_outputs(exported):
    """Se il checkpoint è a due teste, il file espone anche argomento e tipo atteso, coerenti con `probs`."""
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    manifest = json.loads(meta[export_onnx.MANIFEST_KEY])
    if not manifest.get("factored"):
        pytest.skip("checkpoint a classe unica")

    topics, types = json.loads(meta[export_onnx.TOPICS_KEY]), json.loads(meta[export_onnx.TYPES_KEY])
    assert manifest["topics"] == topics and manifest["types"] == types and types[0] == ""
    assert len(manifest["intent_topic"]) == len(manifest["intent_type"]) == len(manifest["labels"])

    ids = np.random.randint(1, 100, (2, 9)).astype(np.int64)
    probs, _, topic_probs, type_probs = session.run(
        [export_onnx.OUTPUT_NAME, export_onnx.NER_OUTPUT_NAME, export_onnx.TOPIC_OUTPUT_NAME, export_onnx.TYPE_OUTPUT_NAME],
        {export_onnx.INPUT_NAME: ids},
    )
    assert topic_probs.shape == (2, len(topics)) and type_probs.shape == (2, len(types))

    # probs è il prodotto normalizzato delle due teste sulle coppie valide
    applies = np.array(manifest["intent_type_applies"], dtype=bool)
    product = topic_probs[:, manifest["intent_topic"]] * np.where(applies, type_probs[:, manifest["intent_type"]], 1.0)
    assert np.allclose(probs, product / product.sum(axis=1, keepdims=True), atol=1e-5)
