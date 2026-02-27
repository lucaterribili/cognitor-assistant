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
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


def predict(sentence_ids):
    with torch.no_grad():
        tokens = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0).to(device)
        output = model(tokens)
        return torch.argmax(output, dim=1).item()


while True:
    user_input = input("Inserisci una frase (o 'esci' per terminare): ")
    if user_input.lower() == 'esci':
        print("Terminato.")
        break

    tokenized_input = model.tokenize(user_input)

    prediction = predict(tokenized_input)

    print(f"Predizione: {intent_dict[str(prediction)]}")
