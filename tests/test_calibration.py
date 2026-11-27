import pytest
import torch

from intellective.calibrate_intent_classifier import calibration_report, fit_temperature
from tests.test_factored_classifier import make_model


def make_underconfident(n=4000, classes=6, seed=0):
    """Logits che indovinano l'80% delle volte ma con confidenza bassa (come dopo il label smoothing)."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=g)
    wrong = torch.rand(n, generator=g) > 0.8
    predicted = torch.where(wrong, (labels + 1) % classes, labels)
    logits = torch.zeros(n, classes)
    logits[torch.arange(n), predicted] = 1.0   # margine piccolo: softmax ~ 0.3
    return logits, labels


def test_fit_temperature_sharpens_an_underconfident_model_without_changing_predictions():
    logits, labels = make_underconfident()

    t = fit_temperature(logits, labels)
    before, after = calibration_report(logits, labels, 1.0), calibration_report(logits, labels, t)

    assert t < 1.0                                   # sotto-sicuro: si rende più netto
    assert after["ece"] < before["ece"] / 3
    assert after["nll"] < before["nll"]
    assert after["accuracy"] == before["accuracy"]   # l'argmax non cambia


def test_fit_temperature_flattens_an_overconfident_model():
    logits, labels = make_underconfident()

    t = fit_temperature(logits * 12, labels)         # troppo sicuro

    assert t > 1.0


def test_temperature_scales_the_probabilities_but_not_the_predicted_intent():
    model, _ = make_model()
    x = torch.randint(1, 50, (5, 6))

    logits, _ = model(x)
    model.temperature = torch.tensor(0.5)
    sharper, _ = model(x)

    assert torch.equal(logits.argmax(-1), sharper.argmax(-1))
    assert torch.softmax(sharper, -1).max(-1).values.mean() > torch.softmax(logits, -1).max(-1).values.mean()


def test_old_checkpoints_without_temperature_still_load():
    model, _ = make_model()
    state = {k: v for k, v in model.state_dict().items() if k != "temperature"}
    fresh, _ = make_model()
    fresh.temperature = torch.tensor(0.9)

    fresh.load_state_dict(state)

    assert float(fresh.temperature) == pytest.approx(0.9)   # senza il campo resta il valore corrente (nessuna calibrazione imposta)


def test_temperature_is_saved_in_the_checkpoint():
    model, _ = make_model()
    model.temperature = torch.tensor(0.75)

    again, _ = make_model()
    again.load_state_dict(model.state_dict())

    assert float(again.temperature) == pytest.approx(0.75)
