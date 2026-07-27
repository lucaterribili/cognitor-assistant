import os

import sentencepiece as spm
import torch

from config import BASE_DIR


_LANGUAGE_ALIASES = {
    "russo": "ru",
    "russian": "ru",
    "ru": "ru",
    "russia": "ru",
    "italiano": "it",
    "italian": "it",
    "it": "it",
    "english": "en",
    "inglese": "en",
    "francese": "fr",
    "spagnolo": "es",
    "tedesco": "de",
    "cinese": "zh",
    "arabo": "ar",
    "giapponese": "ja",
    "portoghese": "pt",
}

_SUPPORTED_TRANSLATION_MODELS = {
    ("it", "ru"): "it-ru",
}

_CACHE: dict[str, tuple[torch.jit.ScriptModule, spm.SentencePieceProcessor]] = {}


def normalize_language(language: str) -> str | None:
    if not language:
        return None
    return _LANGUAGE_ALIASES.get(language.strip().lower(), language.strip().lower())


def resolve_translation_paths(target_language: str, source_language: str = "it") -> tuple[str, str] | None:
    source_code = normalize_language(source_language)
    target_code = normalize_language(target_language)

    if not source_code or not target_code:
        return None

    model_key = (source_code, target_code)
    model_folder = _SUPPORTED_TRANSLATION_MODELS.get(model_key)
    if not model_folder:
        return None

    model_root = os.path.join(BASE_DIR, "models", "translator", model_folder)
    model_path = os.path.join(model_root, "model", "translator.pt")
    tokenizer_path = os.path.join(model_root, "tokenizer", "model.model")

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None

    return model_path, tokenizer_path


def load_translator(target_language: str, source_language: str = "it") -> tuple[torch.jit.ScriptModule, spm.SentencePieceProcessor]:
    paths = resolve_translation_paths(target_language, source_language)
    if paths is None:
        raise FileNotFoundError("Translation model not available for requested language pair")

    model_path, tokenizer_path = paths
    cache_key = f"{model_path}|{tokenizer_path}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    _CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def translate_text(model: torch.jit.ScriptModule, tokenizer: spm.SentencePieceProcessor, src_text: str, max_len: int = 80) -> str:
    src_ids = [tokenizer.bos_id()] + tokenizer.encode_as_ids(src_text) + [tokenizer.eos_id()]
    src_tensor = torch.tensor([src_ids], dtype=torch.long)
    tgt_ids = [tokenizer.bos_id()]

    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long)
            logits = model(src_tensor, tgt_tensor)
            next_token_id = int(logits[0, -1, :].argmax().item())
            tgt_ids.append(next_token_id)
            if next_token_id == tokenizer.eos_id():
                break

    return tokenizer.decode(tgt_ids)
