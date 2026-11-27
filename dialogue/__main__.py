"""
Corpus di conversazioni e modello dei flussi:
    python -m dialogue corpus-check                       # controlla i file in dialogue/corpus/
    python -m dialogue corpus-stats                       # conteggi per decidere cosa aggiungere
    python -m dialogue sample --n 6 --seed 1              # conversazioni campionate dal modello (da rivedere)
    python -m dialogue export-model --out cognitor-flow-model.json    # il modello, per l'engine

Simulatore a regole (solo baseline di confronto, non è la fonte di verità):
    python -m dialogue baseline --n 2000 --seed 99 --noise 0.15
    python -m dialogue generate --n 3000 --seed 1 --noise 0.15 --out dialogues.jsonl
"""
import argparse
import collections
import json
import random
import sys

from dialogue.corpus import load_corpus, stats, validate
from dialogue.flow_model import FlowModel, SiteVocabulary, instantiate
from dialogue.simulator import Simulator, baseline_predictions, write_jsonl
from dialogue.taxonomy import Taxonomy


def corpus_check() -> int:
    conversations, taxonomy = load_corpus(), Taxonomy.load(config_path=None)
    errors = validate(conversations, taxonomy)
    print(f"{len(conversations)} conversazioni, {sum(len(c.turns) for c in conversations)} turni: "
          f"{len(errors)} errori")
    for error in errors:
        print("  -", error)
    return 1 if errors else 0


def corpus_stats() -> None:
    s = stats(load_corpus())
    print(f"{s['conversations']} conversazioni ({s['reviewed']} riviste), {s['turns']} turni, fonti {s['by_source']}")
    for key in ("moves", "actions", "entity_types"):
        print(f"\n{key}:", ", ".join(f"{k} {v}" for k, v in s[key].items()))
    print(f"\nargomenti usati: {len(s['topics'])} (più frequenti: "
          f"{', '.join(f'{k} {v}' for k, v in list(s['topics'].items())[:8])})")
    print("azioni mai viste:", s["actions_never_seen"] or "nessuna", "| mosse mai viste:",
          s["moves_never_seen"] or "nessuna")
    print("\ntransizioni (azione del bot -> mossa dell'utente), le più rare in coda:")
    for k, v in list(s["transitions"].items()):
        print(f"  {v:3}  {k}")


def export_model(out: str) -> None:
    """Scrive il modello (tabelle di conteggi) e qualche metadato del corpus da cui è stato appreso."""
    conversations = load_corpus()
    model = FlowModel.fit(conversations)
    payload = {
        "version": 1,
        "corpus": {"conversations": len(conversations), "turns": sum(len(c.turns) for c in conversations),
                   "reviewed": sum(c.reviewed for c in conversations)},
        "tables": json.loads(model.to_json()),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"modello scritto in {out} ({payload['corpus']['conversations']} conversazioni, "
          f"{payload['corpus']['turns']} turni)")


def sample(n: int, seed: int) -> None:
    conversations, taxonomy = load_corpus(), Taxonomy.load(config_path=None)
    model = FlowModel.fit(conversations)
    vocab = SiteVocabulary.from_taxonomy(taxonomy)
    rng = random.Random(seed)
    for i in range(n):
        print(f"--- conversazione {i + 1}")
        for turn in instantiate(model, model.sample(rng), vocab, rng):
            entities = ", ".join(f"{e['type']}={e['value']}" for e in turn["entities"])
            carry = f" (contesto: {', '.join(turn['carry'])})" if turn["carry"] else ""
            rag = f" [rag {turn['retrieval']}]" if turn["retrieval"] != "ok" else ""
            print(f"  utente  {turn['move']:17} {turn.get('intent') or '':36} {entities}{carry}{rag}")
            print(f"  bot     {turn['bot']}")


def main():
    parser = argparse.ArgumentParser(description="Corpus di conversazioni, modello dei flussi, simulatore")
    parser.add_argument("command", choices=["corpus-check", "corpus-stats", "sample", "export-model", "generate", "baseline"])
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.15, help="errori dell'NLU simulato (0-1)")
    parser.add_argument("--out", default="dialogues.jsonl", help="file di destinazione (generate, export-model)")
    args = parser.parse_args()

    if args.command == "corpus-check":
        sys.exit(corpus_check())
    if args.command == "corpus-stats":
        return corpus_stats()
    if args.command == "sample":
        return sample(args.n, args.seed)
    if args.command == "export-model":
        return export_model(args.out)

    taxonomy = Taxonomy.load()
    simulator = Simulator(taxonomy, noise=args.noise, seed=args.seed)
    if args.command == "generate":
        print(f"{write_jsonl(simulator.generate(args.n), args.out)} turni scritti in {args.out}")
        return

    turns = list(simulator.generate(args.n))
    predicted = baseline_predictions(turns, taxonomy)
    correct = sum(p == t["action"] for p, t in zip(predicted, turns))
    print(f"{len(turns)} turni in {args.n} conversazioni, rumore {args.noise}: baseline a regole {correct / len(turns):.3f}")
    per_action = collections.defaultdict(lambda: [0, 0])
    for p, t in zip(predicted, turns):
        per_action[t["action"]][1] += 1
        per_action[t["action"]][0] += p == t["action"]
    for action, (ok, total) in sorted(per_action.items(), key=lambda kv: -kv[1][1]):
        print(f"  {action:13} {total:6}  ({total / len(turns):5.1%} dei turni)  baseline corretta {ok / total:5.1%}")


if __name__ == "__main__":
    main()
