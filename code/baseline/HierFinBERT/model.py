"""
model.py – Hierarchical FinBERT (classification version)
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class HierarchicalBert(nn.Module):
    """
    Encodes long documents by:
        1. FinBERT on each chunk
        2. Bi-directional LSTM over chunk CLS-embeddings
        3. Linear head → 2-way classification
    """

    # ↓ set to e.g. 4 to fine-tune only the top 4 encoder layers
    _TRAINABLE_LAYERS = 4          # 12 = all layers = no freezing

    def __init__(self, model_name: str, lstm_hidden_size: int,
                 lstm_layers: int, dropout_rate: float = 0.1):
        super().__init__()

        # 1. FinBERT backbone
        self.bert = AutoModel.from_pretrained(model_name)

        # ▸ MEMORY-SAVER #1: checkpointing
        self.bert.gradient_checkpointing_enable()

        # ▸ MEMORY-SAVER #2: optionally freeze lower layers
        if self._TRAINABLE_LAYERS < len(self.bert.encoder.layer):
            for layer in self.bert.encoder.layer[:-self._TRAINABLE_LAYERS]:
                layer.requires_grad_(False)

        # 2. LSTM aggregator
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Classifier head
        self.dropout   = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(lstm_hidden_size * 2, 2)

    def forward(self, input_ids, attention_mask, segment_lengths):
        # ensure (B, S, T) even when B==1
        if input_ids.dim() == 2:
            input_ids      = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        B, S, T = input_ids.shape
        flat_ids      = input_ids.view(-1, T)
        flat_attn     = attention_mask.view(-1, T)

        # FinBERT → CLS embeddings
        cls_vecs = self.bert(flat_ids, attention_mask=flat_attn).pooler_output
        cls_vecs = cls_vecs.view(B, S, -1)

        # pack -> LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            cls_vecs, segment_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(packed)

        # concat final fwd & bwd states
        doc_repr = torch.cat((h_n[-2], h_n[-1]), dim=1)
        logits   = self.regressor(self.dropout(doc_repr))
        return logits
