# models/attention.py
import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=256, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):  # x: (B, L, H)
        B, L, H = x.size()
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, heads, L, L)
        if attention_mask is not None:
            # attention_mask: (B, L) -> (B,1,1,L)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # (B, heads, L, head_dim)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, H)
        return self.out_proj(ctx)
