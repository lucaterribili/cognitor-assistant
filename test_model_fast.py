import json
import os
import torch

from config import TOKENIZER_PATH, BASE_DIR
from intellective.intent_classifier import IntentClassifier
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)

intent_dict_path = os.path.join(BASE_DIR, 'data', 'intent_dict.json')

with open(intent_dict_path, 'r') as f:
    intent_dict = json.load(f)
    intents_number = len(intent_dict)

model_path = os.path.join(BASE_DIR, 'models', 'intent_model_fast.pth')

model = IntentClassifier(
    vocab_size=sp.vocab_size(),
    embed_dim=300,
    hidden_dim=256,
    output_dim=intents_number,
    dropout_prob=0.3,
    sp_model_path=TOKENIZER_PATH,
    fasttext_model_path=os.path.join(BASE_DIR, 'models', 'fasttext_model.bin'),
    freeze_embeddings=True
)

pretrained = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(pretrained, strict=False)



def predict(sentence_ids):
    model.eval()
    with torch.no_grad():
        tokens = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0)
        output = model(tokens)
        return torch.argmax(output, dim=1).item()


while True:
    user_input = input("Inserisci una frase (o 'esci' per terminare): ")
    if user_input.lower() == 'esci':
        print("Terminato.")
        break

    # Tokenizza l'input con SentencePiece
    tokenized_input = sp.EncodeAsIds(user_input)

    # Usa la funzione predict per ottenere la previsione
    prediction = predict(tokenized_input)

    print(f"Predizione: {prediction}")