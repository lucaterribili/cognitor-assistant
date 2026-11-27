"""
Calibra le confidenze dell'Intent Classifier con una sola temperatura (scalatura post-hoc, non cambia le predizioni):
i logits dell'intento si dividono per T, scelta sulla validation minimizzando la log-verosimiglianza negativa.
Il label smoothing del training rende il modello sotto-sicuro (confidenze troppo basse); la temperatura lo corregge.

    python -m intellective.calibrate_intent_classifier        # calibra models/intent_model_fast.pth e lo riscrive

La temperatura è salvata nel checkpoint e finisce nel file ONNX. Si può rilanciare: parte sempre dai logits grezzi.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from agent.model_loader import ModelLoader
from config import BASE_DIR
from intellective.train_intent_classifier import (IntentDataset, collate_fn, expected_calibration_error,
                                                  make_split)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, low: float = 0.2, high: float = 3.0) -> float:
    """T che minimizza la NLL dei logits/T (ricerca a sezione aurea: la NLL è unimodale in T)."""
    nll = lambda t: torch.nn.functional.cross_entropy(logits / t, labels).item()
    ratio = (5 ** 0.5 - 1) / 2
    a, b = low, high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(40):
        if nll(c) < nll(d):
            b, d = d, c
            c = b - ratio * (b - a)
        else:
            a, c = c, d
            d = a + ratio * (b - a)
    return round((a + b) / 2, 4)


def calibration_report(logits, labels, temperature: float) -> dict:
    probs = torch.softmax(logits / temperature, dim=-1)
    confidence, predicted = probs.max(dim=-1)
    return {
        "confidence": confidence.mean().item(),
        "accuracy": (predicted == labels).float().mean().item(),
        "ece": expected_calibration_error([confidence], [predicted == labels]),
        "nll": torch.nn.functional.cross_entropy(logits / temperature, labels).item(),
    }


@torch.no_grad()
def validation_logits(model, dataset, val_idx):
    """Logits grezzi (senza temperatura) e etichette sulla validation."""
    saved, model.temperature = model.temperature.clone(), torch.tensor(1.0)
    model.eval()
    loader = DataLoader(torch.utils.data.Subset(dataset, val_idx.tolist()), batch_size=64, collate_fn=collate_fn)
    logits, labels = [], []
    for inputs, intent_labels, _, _ in loader:
        logits.append(model(inputs)[0])
        labels.append(intent_labels)
    model.temperature = saved
    return torch.cat(logits), torch.cat(labels)


def calibrate(checkpoint: str) -> float:
    loader = ModelLoader(BASE_DIR, torch.device("cpu"))
    loader.model_path = checkpoint
    intent_dict = loader.load_intent_dict()
    model, loaded = loader.load_intent_classifier(loader.load_vocab_size(), len(intent_dict))
    if not loaded:
        raise RuntimeError(f"Checkpoint non caricato: {checkpoint}")

    dataset = IntentDataset(os.path.join(BASE_DIR, ".cognitor", "tokenized_data.npy"))
    _, val_idx, _ = make_split(dataset)
    logits, labels = validation_logits(model, dataset, val_idx)

    temperature = fit_temperature(logits, labels)
    before, after = calibration_report(logits, labels, 1.0), calibration_report(logits, labels, temperature)
    print(f"T = {temperature}: ECE {before['ece']:.3f} -> {after['ece']:.3f}, NLL {before['nll']:.3f} -> "
          f"{after['nll']:.3f}, confidenza media {before['confidence']:.3f} -> {after['confidence']:.3f} "
          f"(accuratezza {after['accuracy']:.3f}, invariata)")

    model.temperature = torch.tensor(temperature)
    torch.save(model.state_dict(), checkpoint)
    return temperature


def main():
    parser = argparse.ArgumentParser(description="Calibra le confidenze dell'Intent Classifier")
    parser.add_argument("--checkpoint", default=os.path.join(BASE_DIR, "models", "intent_model_fast.pth"))
    calibrate(parser.parse_args().checkpoint)


if __name__ == "__main__":
    main()
