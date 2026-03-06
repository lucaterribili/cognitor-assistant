"""
Diagnosi del problema NER
"""
import sys
sys.path.insert(0, '.')

import torch
import json
import os
from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier
import fasttext

def main():
    print("=" * 80)
    print("DIAGNOSI PROBLEMA NER")
    print("=" * 80)

    # Carica configurazione dal modello salvato
    model_path = os.path.join(BASE_DIR, 'models', 'intent_model_fast.pth')
    fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
    intent_dict_path = os.path.join(BASE_DIR, '.cognitor', 'intent_dict.json')

    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato: {model_path}")
        return

    # Carica intent dict per sapere il numero di output
    with open(intent_dict_path, 'r') as f:
        intent_dict = json.load(f)
        num_intents = len(intent_dict)

    # Carica FastText per sapere la vocab size
    ft_model = fasttext.load_model(fasttext_model_path)
    vocab_size = len(ft_model.words)

    vocab_path = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')
    wordvectors_path = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')

    print(f"\n1. Configurazione:")
    print(f"   - Vocab size: {vocab_size}")
    print(f"   - Num intents: {num_intents}")

    # Carica il modello con le dimensioni corrette
    model = IntentClassifier(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=256,
        output_dim=num_intents,
        dropout_prob=0.3,
        wordvectors_path=wordvectors_path,
        vocab_path=vocab_path,
        freeze_embeddings=True
    )

    print(f"   - NER tags: {model.num_ner_tags}")

    # Carica i pesi
    print(f"\n2. Caricamento pesi...")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"   ✓ Modello caricato con successo")

    # Verifica che il CRF sia inizializzato
    print(f"\n3. Verifica componenti NER:")
    print(f"   - CRF presente: {hasattr(model, 'crf')}")
    print(f"   - NER FC layer: {model.ner_fc}")
    print(f"   - CRF num_tags: {model.crf.num_tags}")

    # Test prediction
    print(f"\n4. Test predizione:")
    test_sentences = [
        "prenota un tavolo per due persone a Roma",
        "invia una email a mario@esempio.it",
        "svegliami alle sette domani mattina"
    ]

    for sentence in test_sentences:
        print(f"\n   Input: '{sentence}'")
        result = model.predict(sentence)

        print(f"   Intent: {result['intent_idx']} (conf: {result['intent_confidence']:.3f})")
        print(f"   Intent name: {intent_dict[str(result['intent_idx'])]}")
        print(f"   Tokens: {result['tokens']}")
        print(f"   NER tags: {result['ner_tags']}")
        print(f"   Entities: {result['entities']}")

        if not result['entities']:
            print(f"   ⚠️  NESSUNA ENTITÀ RILEVATA!")

            # Debug: vediamo cosa restituisce il forward
            tokens = result['tokens']
            token_ids = torch.tensor([model.tokenizer.get_word_index(t) for t in tokens]).unsqueeze(0)

            with torch.no_grad():
                intent_logits, ner_predictions = model(token_ids)
                print(f"   Debug - NER predictions raw: {ner_predictions[0]}")
                print(f"   Debug - NER tags decoded: {[model.ner_tag_builder.id2tag[tid] for tid in ner_predictions[0]]}")
        else:
            for ent in result['entities']:
                print(f"      ✓ {ent['entity']}: '{ent['value']}' (pos {ent['start']}-{ent['end']})")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

