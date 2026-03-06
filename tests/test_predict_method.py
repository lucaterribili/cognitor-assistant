"""
Test veloce di test_model_fast.py con un modello dummy
"""
import json
import os
import torch
from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier

# Setup
device = torch.device("cpu")
vocab_path = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')
wordvectors_path = os.path.join(BASE_DIR, '.cognitor', 'wordvectors.vec')

# Carica vocab_size da vocab.json
with open(vocab_path, 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# Carica intent dict
intent_dict_path = os.path.join(BASE_DIR, '.cognitor', 'intent_dict.json')
if os.path.exists(intent_dict_path):
    with open(intent_dict_path, 'r') as f:
        intent_dict = json.load(f)
else:
    # Crea intent dict di test
    intent_dict = {str(i): f"intent_{i}" for i in range(8)}

intents_number = len(intent_dict)

model = IntentClassifier(
    vocab_size=vocab_size,
    embed_dim=300,
    hidden_dim=256,
    output_dim=intents_number,
    dropout_prob=0.3,
    wordvectors_path=wordvectors_path,
    vocab_path=vocab_path,
    freeze_embeddings=True
)
model.to(device)
model.eval()

print("="*80)
print("TEST METODO PREDICT() - INFERENZA SU ENTRAMBE LE TESTE")
print("="*80)

# Test predictions
test_sentences = [
    "prenota tavolo per due a Roma",
    "invia email a mario",
    "svegliami domani alle sette",
]

for sentence in test_sentences:
    print(f"\n{'─'*80}")
    print(f"Input: {sentence}")

    result = model.predict(sentence)

    intent_name = intent_dict[str(result['intent_idx'])]
    confidence = result['intent_confidence']
    entities = result['entities']

    print(f"🎯 Intent: {intent_name} (conf: {confidence:.2%})")

    if entities:
        print(f"🏷️  Entità: {len(entities)} trovate")
        for ent in entities:
            print(f"   • {ent['entity']:12s}: '{ent['value']}'")
    else:
        print(f"🏷️  Nessuna entità")

    print(f"📝 Tokens: {result['tokens']}")
    print(f"📝 Tags: {result['ner_tags']}")

print(f"\n{'='*80}")
print("✅ Test completato! Il metodo predict() funziona su entrambe le teste.")
print("="*80)

