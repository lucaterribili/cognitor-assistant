import torch.nn as nn
import torch
import torch.nn.functional as F
import fasttext
import os
from config import BASE_DIR
from classes.simple_tokenizer import SimpleTokenizer


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attn(x), dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context


class IntentClassifier(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim,
            output_dim,
            dropout_prob,
            fasttext_model_path,
            freeze_embeddings=False,
    ):
        super(IntentClassifier, self).__init__()

        self.tokenizer = SimpleTokenizer(fasttext_model_path)
        self.ft_model = fasttext.load_model(fasttext_model_path)

        embedding_matrix = torch.zeros((vocab_size, embed_dim))
        vocab_tokens = self.ft_model.words[:vocab_size]

        for i, token in enumerate(vocab_tokens):
            embedding_matrix[i] = torch.tensor(self.ft_model.get_word_vector(token))

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
        )

        self.bigru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.bigru(x)
        attn_out = self.attention(gru_out)
        out = self.dropout(attn_out)
        out = self.fc(out)
        return out

    def tokenize(self, text):
        tokens = self.tokenizer(text)
        return [self.tokenizer.get_word_index(token) for token in tokens]
