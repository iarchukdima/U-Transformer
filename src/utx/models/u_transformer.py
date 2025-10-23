from typing import List, Optional

import torch
import torch.nn as nn

from ..modules.transformer_block import TransformerBlock


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, factor: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * factor, out_dim)
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        pad = (self.factor - (T % self.factor)) % self.factor
        if pad:
            x = nn.functional.pad(x, (0, 0, 0, pad))
            T = x.size(1)
        x = x.view(B, T // self.factor, C * self.factor)
        return self.proj(x)


class Upsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, factor: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * factor)
        self.factor = factor

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.proj(x)  # (B, T, out_dim * factor)
        out_dim = x.size(-1) // self.factor
        x = x.view(B, T, self.factor, out_dim).permute(0, 1, 2, 3).contiguous()
        x = x.view(B, T * self.factor, out_dim)
        if x.size(1) > target_len:
            x = x[:, :target_len]
        elif x.size(1) < target_len:
            pad = target_len - x.size(1)
            x = nn.functional.pad(x, (0, 0, 0, pad))
        return x


def calc_heads(embed_dim: int, n_heads_base: int, head_dim: int, strategy: str) -> int:
    if strategy == "constant_heads":
        return max(1, n_heads_base)
    elif strategy == "constant_head_dim":
        return max(1, embed_dim // head_dim)
    else:
        raise ValueError(f"Unknown head strategy: {strategy}")


class UTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        stages: List[int],
        blocks_per_stage: List[int],
        n_heads_base: int,
        head_dim: int,
        head_strategy: str,
        downsample_factor: int = 2,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        use_skips: bool = True,
        tie_weights: bool = False,
    ):
        super().__init__()
        assert len(stages) == len(blocks_per_stage)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.use_skips = use_skips

        # Encoder path
        self.token_emb = nn.Embedding(vocab_size, stages[0])
        self.pos_embs = nn.ModuleList([nn.Embedding(max_seq_len // (downsample_factor ** i) + 2, d) for i, d in enumerate(stages)])
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        current_dim = stages[0]
        self.blocks_per_stage = list(blocks_per_stage)
        for i, (dim, nblocks) in enumerate(zip(stages, self.blocks_per_stage)):
            heads = calc_heads(dim, n_heads_base, head_dim, head_strategy)
            for _ in range(nblocks):
                self.enc_blocks.append(TransformerBlock(dim, heads, mlp_ratio, dropout))
            if i < len(stages) - 1:
                self.downs.append(Downsample(dim, stages[i + 1], downsample_factor))

        # Bottleneck is the last stage
        self.bottleneck_ln = nn.LayerNorm(stages[-1])

        # Decoder path
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(stages) - 1)):
            in_dim = stages[i + 1]
            out_dim = stages[i]
            self.ups.append(Upsample(in_dim, out_dim, downsample_factor))
            heads = calc_heads(out_dim, n_heads_base, head_dim, head_strategy)
            # after upsample, we fuse with skip via addition, same dim
            for _ in range(self.blocks_per_stage[i]):
                self.up_blocks.append(TransformerBlock(out_dim, heads, mlp_ratio, dropout))

        self.final_ln = nn.LayerNorm(stages[0])
        self.head = nn.Linear(stages[0], vocab_size, bias=False)
        if tie_weights:
            # tie to token embedding projected to base dim is not straightforward since dims differ across stages
            # We only support tying when the base stage dim equals token_emb dim
            if self.token_emb.weight.size(1) == stages[0]:
                self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.max_seq_len

        # Encoder
        x = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = x + self.pos_embs[0](pos)

        skips: List[torch.Tensor] = []
        stage_idx = 0
        enc_block_ptr = 0
        for i, nblocks in enumerate(self.blocks_per_stage):
            for _ in range(nblocks):
                x = self.enc_blocks[enc_block_ptr](x)
                enc_block_ptr += 1
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
                stage_idx += 1
                # update positional embedding for new length
                new_T = x.size(1)
                pos = torch.arange(0, new_T, dtype=torch.long, device=idx.device).unsqueeze(0)
                x = x + self.pos_embs[i + 1](pos)

        # Bottleneck
        x = self.bottleneck_ln(x)

        # Decoder
        up_block_ptr = 0
        for i in reversed(range(len(self.downs))):
            target_len = skips[i].size(1)
            x = self.ups[len(self.downs) - 1 - i](x, target_len)
            if self.use_skips:
                x = x + skips[i]
            for _ in range(self.blocks_per_stage[i]):
                x = self.up_blocks[up_block_ptr](x)
                up_block_ptr += 1

        x = self.final_ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # blocks_per_stage stored in self.blocks_per_stage


