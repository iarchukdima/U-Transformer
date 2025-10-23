from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.register_buffer("mask", None, persistent=False)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.mask is None or self.mask.size(0) < seq_len:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            self.mask = mask
        return self.mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # shapes: (B, heads, T, head_dim)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self._causal_mask(T, x.device)
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))
        return y


