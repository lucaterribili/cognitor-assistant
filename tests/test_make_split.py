import numpy as np

from intellective.train_intent_classifier import make_split


class FakeDataset:
    def __init__(self, groups, lengths):
        self.groups, self.lengths = groups, lengths

    def get_group_ids(self):
        return self.groups

    def get_token_lengths(self):
        return self.lengths


def _dataset(single_word_groups=0):
    groups = [f"g{i // 2}" for i in range(40)]            # 20 gruppi da 2 varianti
    lengths = [4] * 40
    for i in range(single_word_groups * 2):
        lengths[i] = 1                                    # i primi gruppi sono di una sola parola
    return FakeDataset(groups, lengths)


def test_without_single_word_groups_the_indices_are_integers():
    train, val, info = make_split(_dataset(0))

    assert train.dtype.kind == "i" and val.dtype.kind == "i"
    assert all(isinstance(i, int) for i in train.tolist())     # Subset(...) li usa come indici
    assert info["single_word_groups"] == 0 and len(train) + len(val) == 40


def test_single_word_groups_are_forced_in_training():
    train, val, info = make_split(_dataset(2))

    assert info["single_word_groups"] == 2
    assert {0, 1, 2, 3} <= set(train.tolist()) and not {0, 1, 2, 3} & set(val.tolist())


def test_variants_of_a_group_stay_on_the_same_side():
    train, val, _ = make_split(_dataset(0))

    groups = _dataset(0).get_group_ids()
    assert not {groups[i] for i in train} & {groups[i] for i in val}
