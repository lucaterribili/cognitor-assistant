import torch

from classes.intent_factorizer import IntentFactorizer
from intellective.intent_classifier import IntentClassifier

ENTITY_TYPES = ["DEPARTMENT", "FACILITY", "DOCUMENT"]
NAMES = ["ask_opening_hours", "ask_department_opening_hours", "ask_facility_opening_hours",
         "ask_phone", "ask_department_phone", "affirm_greeting"]


def make_model(factored=True):
    factors = IntentFactorizer(NAMES, ENTITY_TYPES) if factored else None
    return IntentClassifier(
        vocab_size=50, embed_dim=16, hidden_dim=8, output_dim=len(NAMES), dropout_prob=0.0, intent_factors=factors,
    ).eval(), factors


def test_forward_still_returns_one_logit_per_intent():
    model, _ = make_model()
    x = torch.randint(1, 50, (3, 7))

    intent_logits, ner_predictions = model(x)

    assert intent_logits.shape == (3, len(NAMES))
    assert len(ner_predictions) == 3


def test_joint_probabilities_are_a_distribution_over_the_valid_intents():
    model, _ = make_model()
    x = torch.randint(1, 50, (4, 5))

    intent_logits, _, topic_logits, type_logits = model(x, return_heads=True)
    probs = torch.softmax(intent_logits, dim=-1)

    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert topic_logits.shape == (4, 3) and type_logits.shape == (4, 3)  # 3 argomenti; tipi: nessuno + DEPARTMENT + FACILITY


def test_joint_is_the_normalized_product_of_the_two_heads():
    model, factors = make_model()
    x = torch.randint(1, 50, (2, 6))

    intent_logits, _, topic_logits, type_logits = model(x, return_heads=True)
    topic_p, type_p = torch.softmax(topic_logits, -1), torch.softmax(type_logits, -1)
    type_term = torch.where(model.type_applies.bool(), type_p[:, model.pair_type], torch.ones(1))  # il tipo conta solo se ha senso
    product = topic_p[:, model.pair_topic] * type_term

    assert torch.allclose(torch.softmax(intent_logits, -1), product / product.sum(-1, keepdim=True), atol=1e-5)


def test_predict_exposes_the_heads_only_when_factored():
    factored, _ = make_model(True)
    single, _ = make_model(False)

    assert {"topic_probs", "type_probs"} <= set(factored.predict("orari"))
    assert "topic_probs" not in single.predict("orari")
    assert len(factored.predict("orari")["intent_probs"]) == len(NAMES)


def test_single_head_model_keeps_the_old_structure():
    model, _ = make_model(False)

    assert hasattr(model, "fc") and not hasattr(model, "fc_topic")
    assert "fc.weight" in model.state_dict()


def test_type_head_is_not_trained_nor_used_for_intents_without_a_type():
    import torch.nn as nn

    from intellective.train_intent_classifier import intent_loss

    model, factors = make_model()
    greeting = NAMES.index("affirm_greeting")
    assert factors.type_applies[greeting] is False and factors.type_applies[NAMES.index("ask_phone")] is True

    x = torch.randint(1, 50, (4, 5))
    labels = torch.full((4,), greeting, dtype=torch.long)  # solo saluti: la loss del tipo non deve esistere
    model.train()
    intent_logits, _, topic_logits, type_logits = model(x, ner_tags=torch.zeros(4, 5, dtype=torch.long), return_heads=True)
    loss = intent_loss(model, nn.CrossEntropyLoss(), intent_logits, topic_logits, type_logits, labels)
    loss.backward()

    assert model.fc_type.weight.grad is None or float(model.fc_type.weight.grad.abs().sum()) == 0.0
    assert model.fc_topic.weight.grad is not None and float(model.fc_topic.weight.grad.abs().sum()) > 0.0
