import os
from config import BASE_DIR
from classes.simple_tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()

with open(os.path.join(BASE_DIR, "data/frasi.txt"), "r", encoding="utf-8") as f:
    lines = f.readlines()

with open(os.path.join(BASE_DIR, "data/fast-text-tokenized.txt"), "w", encoding="utf-8") as f:
    for line in lines:
        tokens = tokenizer(line.strip())
        f.write(" ".join(tokens) + "\n")
