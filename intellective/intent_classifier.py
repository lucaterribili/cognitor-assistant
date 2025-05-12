import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

class IntentDataset(Dataset):
    def __init__(self, sentences, labels, max_len=100):
        self.sentences = sentences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # SentencePiece tokenization is assumed to be done externally.
        # Here we assume 'sentence' is already tokenized into IDs.
        if len(sentence) < self.max_len:
            sentence += [1] * (self.max_len - len(sentence))  # Using <PAD> token ID (1)
        else:
            sentence = sentence[:self.max_len]

        return torch.tensor(sentence), torch.tensor(label)

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=2):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out