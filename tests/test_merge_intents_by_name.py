import yaml

from classes.dataset_generator import DatasetGenerator


def test_same_intent_in_two_files_becomes_one_entry_with_all_the_examples():
    base = [{"intent": "ask_phone", "examples": ["numero di telefono", "telefono"]}, {"intent": "ask_email", "examples": ["email"]}]
    site = [{"intent": "ask_phone", "examples": ["telefono", "il numero del [CUP](FACILITY)"]}]

    merged = DatasetGenerator.merge_intents_by_name(base + site)

    assert [i["intent"] for i in merged] == ["ask_phone", "ask_email"]  # ordine della prima comparsa
    assert merged[0]["examples"] == ["numero di telefono", "telefono", "il numero del [CUP](FACILITY)"]  # senza doppioni
    assert merged[1]["examples"] == ["email"]


def test_the_input_is_not_modified_and_other_keys_are_kept():
    items = [{"intent": "a", "examples": ["x"], "note": "n"}, {"intent": "a", "examples": ["y"]}]

    merged = DatasetGenerator.merge_intents_by_name(items)

    assert items[0]["examples"] == ["x"] and len(items) == 2
    assert merged == [{"intent": "a", "examples": ["x", "y"], "note": "n"}]


def test_an_intent_without_examples_does_not_break_the_merge():
    assert DatasetGenerator.merge_intents_by_name([{"intent": "a"}, {"intent": "a", "examples": ["x"]}]) == \
        [{"intent": "a", "examples": ["x"]}]


def test_loading_a_directory_merges_intents_of_different_files(tmp_path, monkeypatch):
    for name, intent in (("01_base.yaml", {"intent": "ask_phone", "examples": ["telefono"]}),
                         ("02_sito.yaml", {"intent": "ask_phone", "examples": ["numero del [CUP](FACILITY)"]})):
        (tmp_path / name).write_text(yaml.dump({"nlu": {"intents": [intent]}}, allow_unicode=True), encoding="utf-8")
    captured = {}
    monkeypatch.setattr(DatasetGenerator, "__init__", lambda self, data: captured.update(data=data))

    DatasetGenerator.load_from_yaml_files([str(tmp_path)])

    intents = captured["data"]["nlu"]["intents"]
    assert len(intents) == 1 and intents[0]["examples"] == ["telefono", "numero del [CUP](FACILITY)"]
