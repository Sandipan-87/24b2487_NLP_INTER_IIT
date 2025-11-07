# models/heads.py
import torch.nn as nn

class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.norm(x)
        return self.classifier(x)  # (B, L, V)

class NSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, cls_hidden):
        return self.classifier(cls_hidden)  # (B, 2)
