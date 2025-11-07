# models/embeddings.py
import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    """
    token + position + segment (token_type) embeddings
    """
    def __init__(self, vocab_size, hidden_size=256, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        bsz, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
