import os
from typing import Callable, List, Tuple

import yaml

from dialogue.actions import Action
from dialogue.state import DialogueState
from dialogue.taxonomy import Taxonomy

DEV_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_set.yaml")


def load_dev_set(path: str = DEV_SET_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["dialogues"]


def evaluate_dev_set(predict: Callable[[dict, DialogueState], str], taxonomy: Taxonomy,
                     path: str = DEV_SET_PATH) -> Tuple[float, List[dict]]:
    """
    `predict(osservazione, stato) -> azione` viene chiamata a ogni turno. Lo stato si aggiorna con l'azione ATTESA
    (come quando si genera, così un errore non sfasa i turni successivi). Restituisce (accuratezza, errori).
    """
    total, misses = 0, []
    for dialogue in load_dev_set(path):
        state = DialogueState()
        for number, turn in enumerate(dialogue["turns"]):
            obs = {"topic": turn["topic"], "topic_conf": turn["topic_conf"], "expected_type": None,
                   "entity_types": turn["entity_types"], "intent_conf": turn["topic_conf"], "rag": turn["rag"]}
            got = predict(obs, state)
            total += 1
            if got != turn["action"]:
                misses.append({"dialogue": dialogue["name"], "turn": number, "expected": turn["action"], "got": got})
            known = turn["topic"] if turn["topic"] and turn["topic_conf"] >= 0.5 else None
            state.update(known, turn["entity_types"], Action(turn["action"]))
    return 1 - len(misses) / total, misses
