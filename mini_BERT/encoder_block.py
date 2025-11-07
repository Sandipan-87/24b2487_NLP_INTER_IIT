# models/encoder_block.py
import torch.nn as nn
from models.attention import MultiHeadSelfAttention

class FeedForward(nn.Module):
    def __init__(self, hidden_size=256, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_size=256, num_heads=4, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ff = FeedForward(hidden_size, ffn_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        x = self.norm1(x + self.attn(x, attention_mask))
        x = self.norm2(x + self.ff(x))
        return x
