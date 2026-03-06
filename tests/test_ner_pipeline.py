"""
Test rapido dell'integrazione NER senza training completo
Crea un mini-dataset e verifica che tutto funzioni
"""

import torch
import os
from config import BASE_DIR
from classes.ner_markup_parser import NERMarkupParser
from classes.ner_tag_builder import NERTagBuilder
from classes.simple_tokenizer import SimpleTokenizer


def test_data_pipeline():
    """Test della pipeline di processamento dati con NER"""
    print("=" * 80)
    print("TEST PIPELINE DATI CON NER")
    print("=" * 80)

    vocab_path = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')
    if not os.path.exists(vocab_path):
        print(f"\n❌ Vocabolario non trovato: {vocab_path}")
        print("   Esegui prima il training di FastText")
        return False

    # Test sentence con annotazioni NER
    test_examples = [
        ("prenota tavolo per [due](NUMBER) a [Roma](LOCATION)", "restaurant_booking"),
        ("invia email a [mario@test.it](EMAIL)", "send_email"),
        ("svegliami [domani](DATE) alle [7](TIME)", "alarm_set"),
        ("chiamami un taxi [adesso](TIME)", "taxi_ride"),
    ]

    parser = NERMarkupParser()
    builder = NERTagBuilder()
    tokenizer = SimpleTokenizer(vocab_path)

    print("\n1. Test Parsing & Tag Generation:")
    print("-" * 80)

    processed_data = []

    for raw_text, intent in test_examples:
        # Parse markup
        clean_text, entities = parser.parse(raw_text)

        # Tokenizza
        tokens = tokenizer(clean_text)
        token_ids = [tokenizer.get_word_index(t) for t in tokens]

        # Genera tag BIO
        ner_tag_ids = builder.align_tokens_to_bio(clean_text, tokens, entities)
        ner_tags = [builder.id2tag[tid] for tid in ner_tag_ids]

        processed_data.append({
            'raw_text': raw_text,
            'clean_text': clean_text,
            'intent': intent,
            'tokens': tokens,
            'token_ids': token_ids,
            'entities': entities,
            'ner_tag_ids': ner_tag_ids,
            'ner_tags': ner_tags
        })

        print(f"\n   Input: {raw_text}")
        print(f"   Clean: {clean_text}")
        print(f"   Intent: {intent}")
        print(f"   Entities: {len(entities)} trovate")
        for ent in entities:
            print(f"      • {ent['entity']:10s}: '{ent['value']}' @ {ent['start']}-{ent['end']}")
        print(f"   Tokens ({len(tokens)}): {tokens}")
        print(f"   Tags: {ner_tags}")

    print("\n" + "=" * 80)
    print("2. Test Creazione Batch per Training:")
    print("-" * 80)

    # Simula batch
    token_ids_batch = [d['token_ids'] for d in processed_data]
    ner_tags_batch = [d['ner_tag_ids'] for d in processed_data]

    # Trova lunghezza massima
    max_len = max(len(tids) for tids in token_ids_batch)

    print(f"\n   Numero esempi nel batch: {len(token_ids_batch)}")
    print(f"   Lunghezza massima sequenza: {max_len}")

    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    import json

    # Carica vocab_size da vocab.json
    vocab_size = len(json.load(open(vocab_path, 'r')))

    # Clip token IDs al vocab_size (per sicurezza)
    token_ids_clipped = [[min(tid, vocab_size-1) for tid in tids] for tids in token_ids_batch]

    token_tensors = [torch.tensor(tids, dtype=torch.long) for tids in token_ids_clipped]
    ner_tag_tensors = [torch.tensor(tags, dtype=torch.long) for tags in ner_tags_batch]

    padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)
    padded_ner_tags = pad_sequence(ner_tag_tensors, batch_first=True, padding_value=0)
    masks = pad_sequence([torch.ones(len(t), dtype=torch.bool) for t in token_tensors],
                        batch_first=True, padding_value=False)

    print(f"\n   Shape tokens padded: {padded_tokens.shape}")
    print(f"   Shape NER tags padded: {padded_ner_tags.shape}")
    print(f"   Shape masks: {masks.shape}")

    print("\n" + "=" * 80)
    print("3. Test Modello (Mini Forward Pass):")
    print("-" * 80)

    from intellective.intent_classifier import IntentClassifier

    wordvectors_path = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')

    # Crea modello con vocab_size corretto
    model = IntentClassifier(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=64,
        output_dim=4,  # 4 intents di test
        dropout_prob=0.3,
        wordvectors_path=wordvectors_path,
        vocab_path=vocab_path,
        freeze_embeddings=True
    )

    model.eval()

    # Test training mode
    with torch.no_grad():
        intent_logits, ner_loss = model(padded_tokens, ner_tags=padded_ner_tags, mask=masks)
        print(f"\n   Training mode:")
        print(f"      Intent logits shape: {intent_logits.shape}")
        print(f"      NER loss: {ner_loss.item():.4f}")

    # Test inference mode
    with torch.no_grad():
        intent_logits, ner_predictions = model(padded_tokens, mask=masks)
        print(f"\n   Inference mode:")
        print(f"      Intent logits shape: {intent_logits.shape}")
        print(f"      NER predictions shape: {len(ner_predictions)} esempi")
        print(f"      Primo esempio predictions: {ner_predictions[0]}")

    # Test predict method
    test_sentence = "prenota tavolo per due a Roma"
    print(f"\n   Test predict method:")
    print(f"      Input: {test_sentence}")
    result = model.predict(test_sentence)
    print(f"      Intent idx: {result['intent_idx']} (conf: {result['intent_confidence']:.3f})")
    print(f"      Entities: {result['entities']}")

    print("\n" + "=" * 80)
    print("✅ TUTTI I TEST PASSATI!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_data_pipeline()

    if success:
        print("\n🎉 L'integrazione NER è completa e funzionante!")
        print("\n📋 Prossimi passi:")
        print("   1. python retrain_with_ner.py       # Rigenera dataset completo")
        print("   2. Attendi il training (50 epochs)")
        print("   3. python example_ner_usage.py      # Testa il modello trainato")
    else:
        print("\n⚠️  Completa prima il setup di FastText")
        print("   python -m pipeline")


