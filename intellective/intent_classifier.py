import torch.nn as nn
import torch
import torch.nn.functional as F
import fasttext
from torchcrf import CRF
from classes.simple_tokenizer import SimpleTokenizer
from classes.ner_tag_builder import NERTagBuilder


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
            num_ner_tags=None,
            freeze_embeddings=False,
    ):
        super(IntentClassifier, self).__init__()

        self.tokenizer = SimpleTokenizer(fasttext_model_path)
        self.ft_model = fasttext.load_model(fasttext_model_path)

        # Inizializza NER tag builder
        self.ner_tag_builder = NERTagBuilder()
        if num_ner_tags is None:
            num_ner_tags = self.ner_tag_builder.num_tags

        embedding_matrix = torch.zeros((vocab_size, embed_dim))
        vocab_tokens = self.ft_model.words[:vocab_size]

        for i, token in enumerate(vocab_tokens):
            embedding_matrix[i] = torch.tensor(self.ft_model.get_word_vector(token))

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
        )

        # BiGRU condiviso per intent e NER
        self.bigru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Branch per Intent Classification
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Branch alternativo per sequenze corte (bypassa attention)
        self.short_seq_fc = nn.Linear(embed_dim, output_dim)
        self.short_seq_dropout = nn.Dropout(dropout_prob)

        # Proiezione per NER su sequenze corte (embed_dim -> hidden_dim*2)
        self.short_seq_ner_proj = nn.Linear(embed_dim, hidden_dim * 2)

        # Branch per NER
        self.ner_dropout = nn.Dropout(dropout_prob)
        self.ner_fc = nn.Linear(hidden_dim * 2, num_ner_tags)
        self.crf = CRF(num_ner_tags, batch_first=True)

        self.num_ner_tags = num_ner_tags

    def forward(self, x, ner_tags=None, mask=None):
        """
        Forward pass che restituisce sia intent logits che NER predictions/loss

        Args:
            x: input token ids [batch_size, seq_len]
            ner_tags: tag NER ground truth [batch_size, seq_len] (opzionale, per training)
            mask: maschera di padding [batch_size, seq_len] (opzionale)

        Returns:
            intent_logits: logits per classificazione intent [batch_size, num_intents]
            ner_output: predictions NER (inference) o loss (training)
        """
        # Salva la forma originale per creare la maschera se necessario
        batch_size, seq_len = x.shape

        # Embedding
        x_embedded = self.embedding(x)

        # GRU per tutte le sequenze (anche quelle di 1 token)
        gru_out, _ = self.bigru(x_embedded)

        # Branch Intent Classification con Attention
        attn_out = self.attention(gru_out)
        intent_out = self.dropout(attn_out)
        intent_logits = self.fc(intent_out)

        # Branch NER
        ner_features = self.ner_dropout(gru_out)
        ner_emissions = self.ner_fc(ner_features)

        if ner_tags is not None:
            # Training: calcola loss con CRF
            if mask is None:
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            ner_loss = -self.crf(ner_emissions, ner_tags, mask=mask, reduction='mean')
            return intent_logits, ner_loss
        else:
            # Inference: decodifica sequenza con Viterbi
            if mask is None:
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            ner_predictions = self.crf.decode(ner_emissions, mask=mask)
            return intent_logits, ner_predictions

    def tokenize(self, text):
        tokens = self.tokenizer(text)
        return [self.tokenizer.get_word_index(token) for token in tokens]

    def predict(self, text):
        """
        Predice intent e entità NER da un testo

        Returns:
            dict con:
                - intent: nome dell'intent predetto
                - intent_confidence: confidenza della predizione
                - entities: lista di entità riconosciute
        """
        self.eval()
        with torch.no_grad():
            # Tokenizza
            tokens = self.tokenizer(text)
            token_ids = torch.tensor([self.tokenizer.get_word_index(t) for t in tokens]).unsqueeze(0)

            # Forward pass
            intent_logits, ner_predictions = self.forward(token_ids)

            # Intent
            intent_probs = F.softmax(intent_logits, dim=-1)
            intent_idx = torch.argmax(intent_probs, dim=-1).item()
            intent_conf = intent_probs[0, intent_idx].item()

            # NER - converti tag ids in etichette e raggruppa entità
            ner_tags = [self.ner_tag_builder.id2tag[tag_id] for tag_id in ner_predictions[0]]
            entities = self._extract_entities(tokens, ner_tags)

            return {
                'intent_idx': intent_idx,
                'intent_confidence': intent_conf,
                'entities': entities,
                'tokens': tokens,
                'ner_tags': ner_tags
            }

    def _extract_entities(self, tokens, ner_tags):
        """
        Estrae entità da tokens e tag BIO

        Returns:
            lista di dict con start, end, entity, value
        """
        entities = []
        current_entity = None

        for idx, (token, tag) in enumerate(zip(tokens, ner_tags)):
            if tag.startswith('B-'):
                # Salva entità precedente se esiste
                if current_entity:
                    entities.append(current_entity)
                # Inizia nuova entità
                entity_type = tag[2:]
                current_entity = {
                    'start': idx,
                    'end': idx + 1,
                    'entity': entity_type,
                    'value': token
                }
            elif tag.startswith('I-') and current_entity:
                # Continua entità corrente
                entity_type = tag[2:]
                if entity_type == current_entity['entity']:
                    current_entity['end'] = idx + 1
                    current_entity['value'] += ' ' + token
            else:
                # Tag O o fine entità
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Aggiungi ultima entità se presente
        if current_entity:
            entities.append(current_entity)

        return entities

