import argparse
from pathlib import Path
from typing import Tuple

import torch

from .config import load_config, Config
from .models.gpt_small import GPTSmall
from .models.u_transformer import UTransformer


def build_model(cfg: Config) -> torch.nn.Module:
    # Prefer vocab sizes defined in model configs to avoid data/tokenizer dependencies
    if cfg.model_type == "gpt":
        vocab_size = cfg.gpt.vocab_size
        return GPTSmall(
            vocab_size=vocab_size,
            max_seq_len=cfg.data.max_seq_len,
            n_layers=cfg.gpt.n_layers,
            n_heads=cfg.gpt.n_heads,
            embed_dim=cfg.gpt.embed_dim,
            mlp_ratio=cfg.gpt.mlp_ratio,
            dropout=cfg.gpt.dropout,
        )
    else:
        vocab_size = cfg.u.vocab_size
        return UTransformer(
            vocab_size=vocab_size,
            max_seq_len=cfg.data.max_seq_len,
            stages=cfg.u.stages,
            blocks_per_stage=cfg.u.blocks_per_stage,
            n_heads_base=cfg.u.n_heads_base,
            head_dim=cfg.u.head_dim,
            head_strategy=cfg.u.head_strategy,
            downsample_factor=cfg.u.downsample_factor,
            dropout=cfg.u.dropout,
            use_skips=cfg.u.use_skips,
        )


def count_params(model: torch.nn.Module) -> Tuple[int, int, int]:
    total = 0
    trainable = 0
    bytes_total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        bytes_total += n * p.element_size()
    return total, trainable, bytes_total


def human_bytes(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(nbytes)
    for u in units:
        if size < 1024.0:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs_glob", type=str, default="configs/*.yaml", help="Glob to config files relative to project root")
    args = ap.parse_args()

    # Resolve project root (two levels up from this file: src/utx/... -> project)
    root_dir = Path(__file__).resolve().parents[2]
    cfg_paths = sorted((root_dir / args.configs_glob).parent.glob((root_dir / args.configs_glob).name))
    if not cfg_paths:
        print(f"No configs matched: {args.configs_glob}")
        return

    print("model_name, model_type, total_params, trainable_params, param_memory")
    for cfg_path in cfg_paths:
        print(f"Processing {cfg_path.name}")
        try:
            cfg = load_config(str(cfg_path))
            model = build_model(cfg)
            total, trainable, bytes_total = count_params(model)
            print(f"{cfg_path.name}, {cfg.model_type}, {total:,}, {trainable:,}, {human_bytes(bytes_total)}")
        except Exception as e:
            print(f"{cfg_path.name}, ERROR: {e}")


if __name__ == "__main__":
    main()


