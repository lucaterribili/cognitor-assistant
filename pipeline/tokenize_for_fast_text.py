from sentencepiece import SentencePieceProcessor

sp = SentencePieceProcessor()
sp.Load("models/sentencepiece.model")

with open("data/frasi.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("data/fast-text-tokenized.txt", "w", encoding="utf-8") as f:
    for line in lines:
        tokens = sp.EncodeAsPieces(line.strip())
        f.write(" ".join(tokens) + "\n")