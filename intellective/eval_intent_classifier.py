"""
Accuratezza di un checkpoint dell'Intent Classifier sulla validation del dataset attuale
(stesso split group-aware del training, quindi confrontabile tra checkpoint).

    python -m intellective.eval_intent_classifier                              # models/intent_model_fast.pth
    python -m intellective.eval_intent_classifier --checkpoint models/backup-678/intent_model_fast.pth

Riporta `intent` (intento completo), `topic` (argomento) e `type` (tipo di entità atteso). Per un modello a classe
unica argomento e tipo sono ricavati dall'intento predetto, quindi i numeri sono confrontabili col modello a due teste.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from agent.model_loader import ModelLoader
from classes.intent_factorizer import IntentFactorizer
from classes.ner_tag_builder import NERTagBuilder
from config import BASE_DIR
from intellective.train_intent_classifier import IntentDataset, collate_fn, evaluate_accuracy, make_split


def evaluate_checkpoint(checkpoint: str) -> dict:
    device = torch.device("cpu")
    loader = ModelLoader(BASE_DIR, device)
    loader.model_path = checkpoint
    intent_dict = loader.load_intent_dict()
    model, loaded = loader.load_intent_classifier(loader.load_vocab_size(), len(intent_dict))
    if not loaded:
        raise RuntimeError(f"Checkpoint non caricato: {checkpoint}")

    if not hasattr(model, "pair_topic"):  # modello a classe unica: i fattori servono solo per misurare
        factors = IntentFactorizer.from_intent_dict(intent_dict, NERTagBuilder().ENTITY_TYPES)
        model.register_buffer("pair_topic", torch.tensor(factors.pair_topic, dtype=torch.long))
        model.register_buffer("pair_type", torch.tensor(factors.pair_type, dtype=torch.long))

    dataset = IntentDataset(os.path.join(BASE_DIR, ".cognitor", "tokenized_data.npy"))
    _, val_idx, _ = make_split(dataset)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx.tolist()), batch_size=64, collate_fn=collate_fn)
    return evaluate_accuracy(model, val_loader, device)


def main():
    parser = argparse.ArgumentParser(description="Accuratezza di un checkpoint sulla validation")
    parser.add_argument("--checkpoint", default=os.path.join(BASE_DIR, "models", "intent_model_fast.pth"))
    result = evaluate_checkpoint(parser.parse_args().checkpoint)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}))


if __name__ == "__main__":
    main()
