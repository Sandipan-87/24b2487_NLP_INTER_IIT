# models/bert_encoder.py
import torch.nn as nn
from models.embeddings import BertEmbeddings
from models.encoder_block import EncoderBlock

class MiniBERT(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, num_heads=4, ffn_dim=512,
                 max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(hidden_size, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x  # (B, L, H)
