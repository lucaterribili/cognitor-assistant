"""
Script diagnostico per capire se FastText è il problema con le sequenze corte
"""
import json
import os
import torch
import torch.nn.functional as F
import fasttext
import numpy as np
from config import BASE_DIR
from intellective.intent_classifier import IntentClassifier

device = torch.device("cpu")
print("="*80)
print("DIAGNOSI FASTTEXT - PROBLEMA SEQUENZE CORTE")
print("="*80)

# Carica modello FastText
fasttext_model_path = os.path.join(BASE_DIR, 'models', 'fasttext_model.bin')
ft_model = fasttext.load_model(fasttext_model_path)
vocab_size = len(ft_model.words)

print(f"\n📊 Info FastText:")
print(f"   • Vocabolario: {vocab_size} parole")
print(f"   • Dimensione embedding: {ft_model.get_dimension()}")

# Carica intent classifier trainato
intent_dict_path = os.path.join(BASE_DIR, 'data', 'intent_dict.json')
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
    freeze_embeddings=True
)

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Modello trainato caricato")
except Exception as e:
    print(f"⚠️  Usando modello non trainato: {e}")

model.to(device)
model.eval()

# Test cases: confronto sequenze corte vs lunghe
test_cases = [
    # Sequenze CORTE (1-2 token)
    "ciao",
    "ok",
    "sì",
    "no",
    "aiuto",
    "grazie",
    "stop",
    "annulla",

    # Sequenze MEDIE (3-5 token)
    "qual è la capitale",
    "come stai oggi",
    "che ore sono",

    # Sequenze LUNGHE (6+ token)
    "dammi informazioni sulla capitale d'Italia",
    "vorrei sapere che ore sono adesso per favore",
    "puoi dirmi il meteo di domani a Roma",
]

print(f"\n{'='*80}")
print("TEST 1: EMBEDDINGS FASTTEXT")
print("="*80)

for text in ["ciao", "ok", "vorrei sapere che ore sono"]:
    tokens = model.tokenizer(text)
    print(f"\n📝 Frase: '{text}'")
    print(f"   Tokens: {tokens} (len={len(tokens)})")

    # Estrai embeddings da FastText
    embeddings = []
    for token in tokens:
        vec = ft_model.get_word_vector(token)
        embeddings.append(vec)
        print(f"   • '{token}': norm={np.linalg.norm(vec):.4f}, mean={np.mean(vec):.4f}, std={np.std(vec):.4f}")

    if len(embeddings) > 1:
        # Calcola similarità tra token
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        print(f"   Similarità primi 2 token: {similarity:.4f}")

print(f"\n{'='*80}")
print("TEST 2: PREDIZIONI MODELLO PER LUNGHEZZA SEQUENZA")
print("="*80)

results_by_length = {}

for text in test_cases:
    tokens = model.tokenizer(text)
    seq_len = len(tokens)

    result = model.predict(text)
    intent_name = intent_dict[str(result['intent_idx'])]
    confidence = result['intent_confidence']

    if seq_len not in results_by_length:
        results_by_length[seq_len] = []

    results_by_length[seq_len].append({
        'text': text,
        'intent': intent_name,
        'confidence': confidence,
        'tokens': tokens
    })

    print(f"\n📝 '{text}' ({seq_len} token)")
    print(f"   🎯 Intent: {intent_name}")
    print(f"   📊 Confidenza: {confidence:.2%}")
    print(f"   🔤 Tokens: {tokens}")

print(f"\n{'='*80}")
print("TEST 3: ANALISI STATISTICA PER LUNGHEZZA")
print("="*80)

for seq_len in sorted(results_by_length.keys()):
    results = results_by_length[seq_len]
    confidences = [r['confidence'] for r in results]
    avg_conf = np.mean(confidences)
    std_conf = np.std(confidences)

    print(f"\n📊 Sequenze di {seq_len} token(s):")
    print(f"   • N° esempi: {len(results)}")
    print(f"   • Confidenza media: {avg_conf:.2%}")
    print(f"   • Deviazione standard: {std_conf:.4f}")
    print(f"   • Min confidenza: {min(confidences):.2%}")
    print(f"   • Max confidenza: {max(confidences):.2%}")

print(f"\n{'='*80}")
print("TEST 4: ANALISI LAYER-BY-LAYER (sequenza corta vs lunga)")
print("="*80)

# Test dettagliato: confronto layer-by-layer
short_text = "ciao"
long_text = "dammi informazioni sulla capitale"

with torch.no_grad():
    for text in [short_text, long_text]:
        print(f"\n📝 Analisi: '{text}'")
        tokens = model.tokenizer(text)
        token_ids = torch.tensor([model.tokenizer.get_word_index(t) for t in tokens]).unsqueeze(0).to(device)

        print(f"   Tokens: {tokens} (len={len(tokens)})")
        print(f"   Token IDs: {token_ids.squeeze().tolist()}")

        # Embedding layer
        x_embedded = model.embedding(token_ids)
        print(f"   Embedding shape: {x_embedded.shape}")
        print(f"   Embedding norm: {x_embedded.norm().item():.4f}")
        print(f"   Embedding mean: {x_embedded.mean().item():.4f}")
        print(f"   Embedding std: {x_embedded.std().item():.4f}")

        # GRU layer
        gru_out, _ = model.bigru(x_embedded)
        print(f"   GRU output shape: {gru_out.shape}")
        print(f"   GRU norm: {gru_out.norm().item():.4f}")
        print(f"   GRU mean: {gru_out.mean().item():.4f}")
        print(f"   GRU std: {gru_out.std().item():.4f}")

        # Attention layer
        attn_out = model.attention(gru_out)
        print(f"   Attention output shape: {attn_out.shape}")
        print(f"   Attention norm: {attn_out.norm().item():.4f}")
        print(f"   Attention mean: {attn_out.mean().item():.4f}")

        # Intent logits
        intent_out = model.dropout(attn_out)
        intent_logits = model.fc(intent_out)
        intent_probs = F.softmax(intent_logits, dim=-1)

        print(f"   Intent logits shape: {intent_logits.shape}")
        print(f"   Intent logits (top 3): {intent_logits.squeeze()[:3].tolist()}")
        print(f"   Max probabilità: {intent_probs.max().item():.2%}")
        print(f"   Entropy: {-(intent_probs * torch.log(intent_probs + 1e-10)).sum().item():.4f}")

print(f"\n{'='*80}")
print("CONCLUSIONI")
print("="*80)

