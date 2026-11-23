"""
Script CLI per testare esclusivamente la classificazione dell'Intent e l'estrazione delle entità NER,
senza avviare la gestione della conversazione (no dialogue policy, no sessioni, no rules/responses).

Uso:
  # Modalità interattiva (loop da terminale):
  python scripts/test_nlu.py

  # Modalità singola frase:
  python scripts/test_nlu.py "a che ora apre l'ecocentro comunale?"
  python scripts/test_nlu.py --top 5 "come giustifico l'assenza sul registro elettronico"
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agent.agent import Agent


def print_prediction(text: str, prediction: dict, intent_dict: dict, top_k: int = 3):
    print("\n" + "=" * 60)
    print(f"Testo: \"{text}\"")
    print("-" * 60)
    intent = prediction.get("intent")
    confidence = prediction.get("confidence", 0.0)
    print(f"🎯 INTENT PREDETTO : {intent} ({confidence:.2%})")

    probs = prediction.get("intent_probs")
    if probs is not None and len(probs) > 0:
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"\n📊 Top {top_k} probabilità:")
        for idx, prob in indexed:
            name = intent_dict.get(str(idx), f"id_{idx}")
            mark = "👉" if name == intent else "  "
            print(f"   {mark} {name:<30} {prob:.2%}")

    entities = prediction.get("entities", [])
    print("\n🏷️  ENTITÀ NER:")
    if entities:
        for e in entities:
            print(f"   • [{e['entity']}] \"{e['value']}\" (span: {e['start']}..{e['end']})")
    else:
        print("   (Nessuna entità estratta)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Testa Intent e NER (senza gestione conversazione)")
    parser.add_argument("query", nargs="*", help="Frase da analizzare (se omessa, avvia la modalità interattiva)")
    parser.add_argument("--top", type=int, default=3, help="Numero di alternative intent da mostrare (default: 3)")
    args = parser.parse_args()

    agent = Agent()
    loaded = agent.load_models()
    if not loaded:
        print("❌ Impossibile caricare il modello addestrato. Assicurati che i modelli siano in models/ e .cognitor/")
        sys.exit(1)

    if args.query:
        query_text = " ".join(args.query).strip()
        prediction = agent.predict(query_text)
        print_prediction(query_text, prediction, agent.intent_dict, top_k=args.top)
    else:
        print("\n" + "=" * 60)
        print("COGNITOR NLU TESTER (Solo Intent & NER)")
        print("Scrivi una frase per testarla, oppure 'esci'/'q' per terminare.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nTesto > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nUscita.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("esci", "exit", "quit", "q"):
                print("Arrivederci!")
                break

            prediction = agent.predict(user_input)
            print_prediction(user_input, prediction, agent.intent_dict, top_k=args.top)


if __name__ == "__main__":
    main()
