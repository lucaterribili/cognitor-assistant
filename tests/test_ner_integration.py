"""
Test di integrazione per verificare il funzionamento del NER nel modello IntentClassifier
"""

import torch
import os
from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier
from classes.ner_markup_parser import NERMarkupParser
from classes.ner_tag_builder import NERTagBuilder

def test_ner_components():
    print("=" * 80)
    print("Test NER Components")
    print("=" * 80)

    # Test NERMarkupParser
    parser = NERMarkupParser()
    test_text = "prenota un tavolo per [due](NUMBER) persone a [Roma](LOCATION)"
    clean_text, entities = parser.parse(test_text)

    print(f"\n1. NER Markup Parser")
    print(f"   Input: {test_text}")
    print(f"   Clean text: {clean_text}")
    print(f"   Entities: {entities}")

    # Test NERTagBuilder
    builder = NERTagBuilder()
    print(f"\n2. NER Tag Builder")
    print(f"   Numero di tag: {builder.num_tags}")
    print(f"   Tag disponibili: {list(builder.tag2id.keys())}")

    # Test allineamento token -> BIO tags
    from classes.simple_tokenizer import SimpleTokenizer
    fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

    if os.path.exists(fasttext_model_path):
        tokenizer = SimpleTokenizer(fasttext_model_path)
        tokens = tokenizer(clean_text)
        tag_ids = builder.align_tokens_to_bio(clean_text, tokens, entities)
        tags = [builder.id2tag[tid] for tid in tag_ids]

        print(f"\n3. Token Alignment")
        print(f"   Tokens: {tokens}")
        print(f"   Tags: {tags}")

        for token, tag in zip(tokens, tags):
            print(f"      {token:15s} -> {tag}")
    else:
        print(f"\n   ⚠ FastText model non trovato in {fasttext_model_path}")
        print("   Esegui prima il training di FastText")


def test_model_structure():
    print("\n" + "=" * 80)
    print("Test Model Structure")
    print("=" * 80)

    fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

    if not os.path.exists(fasttext_model_path):
        print(f"⚠ FastText model non trovato in {fasttext_model_path}")
        return

    # Crea un modello di test
    model = IntentClassifier(
        vocab_size=10000,
        embed_dim=300,
        hidden_dim=128,
        output_dim=8,  # numero di intent
        dropout_prob=0.3,
        fasttext_model_path=fasttext_model_path,
        freeze_embeddings=True
    )

    print(f"\n1. Architettura Modello:")
    print(f"   - Intent output dim: {model.fc.out_features}")
    print(f"   - NER tags: {model.num_ner_tags}")
    print(f"   - Hidden dim: 128 (bidirezionale -> 256)")

    # Test forward pass
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_ner_tags = torch.randint(0, model.num_ner_tags, (batch_size, seq_len))
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    print(f"\n2. Test Forward Pass (Training mode):")
    intent_logits, ner_loss = model(dummy_input, ner_tags=dummy_ner_tags, mask=dummy_mask)
    print(f"   - Intent logits shape: {intent_logits.shape}")
    print(f"   - NER loss: {ner_loss.item():.4f}")

    print(f"\n3. Test Forward Pass (Inference mode):")
    intent_logits, ner_predictions = model(dummy_input)
    print(f"   - Intent logits shape: {intent_logits.shape}")
    print(f"   - NER predictions (primo esempio): {ner_predictions[0][:5]}...")

    print(f"\n✅ Test del modello completato con successo!")


def test_prediction():
    print("\n" + "=" * 80)
    print("Test Prediction")
    print("=" * 80)

    model_path = os.path.join(BASE_DIR, 'models', 'intent_model_fast.pth')
    fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')

    if not os.path.exists(model_path):
        print(f"⚠ Modello non trovato in {model_path}")
        print("   Esegui prima il training del modello")
        return

    if not os.path.exists(fasttext_model_path):
        print(f"⚠ FastText model non trovato in {fasttext_model_path}")
        return

    # Carica il modello
    model = IntentClassifier(
        vocab_size=10000,
        embed_dim=300,
        hidden_dim=256,
        output_dim=8,
        dropout_prob=0.3,
        fasttext_model_path=fasttext_model_path,
        freeze_embeddings=True
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Test prediction
    test_sentences = [
        "prenota un tavolo per due persone a Roma",
        "invia una email a mario@esempio.it",
        "svegliami alle sette domani mattina",
        "chiamami un taxi adesso"
    ]

    print("\nPredizioni:")
    for sentence in test_sentences:
        result = model.predict(sentence)
        print(f"\n   Input: {sentence}")
        print(f"   Intent: {result['intent_idx']} (conf: {result['intent_confidence']:.3f})")
        print(f"   Entities: {result['entities']}")
        if result['entities']:
            for ent in result['entities']:
                print(f"      - {ent['entity']}: '{ent['value']}' (pos {ent['start']}-{ent['end']})")


if __name__ == "__main__":
    test_ner_components()
    test_model_structure()
    test_prediction()

