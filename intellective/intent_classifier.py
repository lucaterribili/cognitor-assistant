import torch.nn as nn
import torch
import torch.nn.functional as F
import fasttext
from sentencepiece import SentencePieceProcessor  # Per il tokenizer


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
            sp_model_path,  # Percorso al modello SentencePiece
            fasttext_model_path,  # Percorso al modello FastText (.bin o .vec)
            freeze_embeddings=False,  # Se congelare gli embedding durante il training
    ):
        super(IntentClassifier, self).__init__()

        # Carica SentencePiece
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)
        # Carica FastText
        self.ft_model = fasttext.load_model(fasttext_model_path)

        # Inizializza la matrice di embedding con FastText
        embedding_matrix = torch.zeros((vocab_size, embed_dim))
        for i in range(vocab_size):
            token = self.sp_model.IdToPiece(i)
            embedding_matrix[i] = torch.tensor(self.ft_model.get_word_vector(token))

        # Crea il layer di embedding
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,  # Se True, gli embedding non vengono aggiornati
        )

        # Resto del modello
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