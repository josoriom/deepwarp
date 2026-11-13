import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepwarp.operations import layer_norm

blocked = -1e9


class IsomerAttention(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.heads = heads
        self.head_size = size // heads
        self.norm_1 = nn.LayerNorm(size)
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)
        self.output = nn.Linear(size, size)
        self.norm_2 = nn.LayerNorm(size)
        self.linear_1 = nn.Linear(size, size)
        self.linear_2 = nn.Linear(size, size)

    def split_heads(self, values):
        batch, targets, _ = values.shape
        return values.reshape(batch, targets, self.heads, self.head_size).transpose(1, 2)

    def attend(self, hidden, same_trace):
        batch, targets, size = hidden.shape
        query = self.split_heads(self.query(hidden))
        key = self.split_heads(self.key(hidden))
        value = self.split_heads(self.value(hidden))
        penalty = torch.where(same_trace, 0.0, blocked)
        scores = query @ key.transpose(2, 3) / math.sqrt(self.head_size) + penalty
        shared = torch.softmax(scores, dim=-1) @ value
        return self.output(shared.transpose(1, 2).reshape(batch, targets, size))

    def forward(self, hidden, same_trace):
        hidden = hidden + self.attend(layer_norm(self.norm_1, hidden), same_trace)
        return hidden + self.linear_2(F.gelu(self.linear_1(layer_norm(self.norm_2, hidden))))
