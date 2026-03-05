import json
import os
import torch
import fasttext

from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
ft_model = fasttext.load_model(fasttext_model_path)
vocab_size = len(ft_model.words)

vocab_path = os.path.join(BASE_DIR, '.cognitor', 'vocab.json')
intent_dict_path = os.path.join(BASE_DIR, '.cognitor', 'intent_dict.json')

with open(intent_dict_path, 'r') as f:
    intent_dict = json.load(f)
    intents_number = len(intent_dict)

model_path = os.path.join(BASE_DIR, 'models', 'intent_model_fast.pth')

model = IntentClassifier(
    vocab_size=vocab_size,
    embed_dim=300,
    hidden_dim=256,
    output_dim=intents_number,
    dropout_prob=0.3,
    fasttext_model_path=fasttext_model_path,
    vocab_path=vocab_path,
    freeze_embeddings=True
)

# Prova a caricare il modello trainato
model_loaded = False
try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Modello caricato da: {model_path}")
    model_loaded = True
except FileNotFoundError:
    print(f"⚠️  File modello non trovato: {model_path}")
    print(f"⚠️  Usando modello NON trainato (pesi random)")
    print(f"⚠️  Esegui 'python -m pipeline' per trainare il modello con NER")
except RuntimeError as e:
    print(f"⚠️  Errore caricamento modello (probabilmente struttura incompatibile)")
    print(f"⚠️  Dettaglio: {str(e)[:150]}...")
    print(f"⚠️  Usando modello NON trainato (pesi random)")
    print(f"⚠️  Il modello vecchio non è compatibile. Training in corso con 'python -m pipeline'")
except Exception as e:
    print(f"⚠️  Errore imprevisto: {e}")
    print(f"⚠️  Usando modello NON trainato (pesi random)")

model.to(device)
model.eval()

print("\n" + "="*80)
print("TEST INTENT CLASSIFIER CON NER")
print("="*80)
print(f"\nModello:")
print(f"  • Stato: {'✓ TRAINATO' if model_loaded else '⚠️  NON TRAINATO (pesi random)'}")
print(f"  • Intents: {intents_number}")
print(f"  • NER tags: {model.num_ner_tags}")
print(f"  • Device: {device}")

if not model_loaded:
    print(f"\n⚠️  ATTENZIONE: Il modello usa pesi casuali!")
    print(f"   Le predizioni non saranno accurate.")
    print(f"   Esegui 'python -m pipeline' per trainare.\n")
else:
    print(f"\n✓ Modello pronto per inferenza accurata\n")

print("Inserisci una frase (o 'esci' per terminare)\n")

while True:
    user_input = input("> ").strip()
    if user_input.lower() in ['esci', 'exit', 'quit', 'q']:
        print("\n👋 Terminato.")
        break

    if not user_input:
        continue

    # Usa il metodo predict() che gestisce sia intent che NER
    result = model.predict(user_input)

    intent_name = intent_dict[str(result['intent_idx'])]
    confidence = result['intent_confidence']
    entities = result['entities']

    print(f"\n{'─'*80}")
    print(f"🎯 Intent: {intent_name} (confidenza: {confidence:.2%})")

    if entities:
        print(f"🏷️  Entità riconosciute:")
        for ent in entities:
            print(f"   • {ent['entity']:12s}: '{ent['value']}'")
    else:
        print(f"🏷️  Nessuna entità riconosciuta")

    # Debug info (opzionale)
    print(f"\n📝 Debug:")
    print(f"   Tokens: {' | '.join(result['tokens'])}")
    print(f"   Tags:   {' | '.join(result['ner_tags'])}")
    print(f"{'─'*80}\n")
