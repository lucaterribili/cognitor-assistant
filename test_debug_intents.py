"""
Test per verificare che i logits e le probabilità vengano restituiti correttamente
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent import Agent


def test_intent_debug():
    """Test per verificare il debug degli intent"""
    print("Caricamento modello...")
    agent = Agent()
    agent.load_models()
    agent.load_knowledge()

    # Test con alcuni input
    test_inputs = [
        "Ciao come stai?",
        "Che tempo fa oggi?",
        "Apri spotify"
    ]

    for text in test_inputs:
        print("\n" + "="*60)
        print(f"Input: {text}")
        print("="*60)

        prediction = agent.predict(text)

        print(f"\nIntent predetto: {prediction['intent']}")
        print(f"Confidenza: {prediction['confidence']:.4f}")
        print(f"Entità: {prediction.get('entities', [])}")

        # Verifica che i logits e le probabilità siano presenti
        if 'intent_probs' in prediction and prediction['intent_probs']:
            intent_probs = prediction['intent_probs']
            intent_logits = prediction.get('intent_logits', [])

            # Ordina per probabilità discendente
            sorted_indices = sorted(
                range(len(intent_probs)),
                key=lambda i: intent_probs[i],
                reverse=True
            )

            # Filtra solo intent con probabilità significativa (> 0.0001) e limita a 5
            significant_intents = [
                idx for idx in sorted_indices
                if intent_probs[idx] > 0.0001
            ][:5]  # Massimo 5

            if significant_intents:
                print(f"\n[DEBUG] Top {len(significant_intents)} Intent (prob > 0.0001):")
                for rank, idx in enumerate(significant_intents, 1):
                    intent_name = agent.intent_dict.get(str(idx), f"unknown_{idx}")
                    prob = intent_probs[idx]
                    logit = intent_logits[idx] if idx < len(intent_logits) else 0.0
                    print(f"  {rank}. {intent_name}: prob={prob:.4f}, logit={logit:.4f}")
            else:
                print("\n[DEBUG] Nessun intent con probabilità > 0.0001")
        else:
            print("\n[ERRORE] Logits e probabilità non presenti nella risposta!")


if __name__ == "__main__":
    test_intent_debug()



