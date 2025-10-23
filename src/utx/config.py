from dataclasses import dataclass, field
from typing import List, Optional, Literal
import yaml


@dataclass
class DataConfig:
    file: str = "input.txt"
    train_split: float = 0.95
    max_seq_len: int = 512
    shuffle: bool = True
    # data source: local file or Hugging Face Wikipedia stream
    source: Literal["file", "hf_wikipedia"] = "file"
    wiki_config: str = "20220301.en"  # e.g., 20220301.en, 20220301.simple
    streaming: bool = True
    # removed window stride; using dense windowing by default


@dataclass
class TokenizerConfig:
    type: Literal["byte", "hf"] = "byte"
    hf_name: str = "gpt2"
    cache_dir: Optional[str] = None
    add_special_tokens: bool = True


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 200


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "cuda"
    batch_size: int = 32
    grad_accum_steps: int = 1
    epochs: int = 2
    eval_interval: int = 200
    ckpt_dir: str = "checkpoints"
    amp: bool = True
    # for streaming datasets where len(loader) is undefined
    steps_per_epoch: Optional[int] = None
    val_batches: int = 200
    # new training controls
    grad_clip: Optional[float] = 1.0
    ema_decay: Optional[float] = 0.999
    label_smoothing: float = 0.0
    early_stop_patience: int = 5
    # lr scheduler options
    scheduler: Literal["cosine", "plateau"] = "cosine"
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    # curriculum: max seq len per epoch (if provided, overrides data.max_seq_len per epoch)
    curriculum_max_seq: Optional[list] = None


@dataclass
class GPTConfig:
    vocab_size: int = 259  # 256 byte tokens + BOS/EOS/PAD
    n_layers: int = 8
    n_heads: int = 8
    embed_dim: int = 512
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    tie_weights: bool = False


HeadStrategy = Literal["constant_heads", "constant_head_dim"]


@dataclass
class UTransformerConfig:
    vocab_size: int = 259
    stages: List[int] = field(default_factory=lambda: [384, 256, 160])
    # stages are per-level embedding dims going down; the last is the bottleneck dim
    blocks_per_stage: List[int] = field(default_factory=lambda: [2, 2, 4])
    n_heads_base: int = 6
    head_dim: int = 64
    head_strategy: HeadStrategy = "constant_heads"
    dropout: float = 0.0
    # down/up sampling factor per stage for sequence length (e.g., 2 means halving length)
    downsample_factor: int = 2
    use_skips: bool = True
    tie_weights: bool = False


@dataclass
class Config:
    model_type: Literal["gpt", "u_transformer"] = "gpt"
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    u: UTransformerConfig = field(default_factory=UTransformerConfig)


def load_config(path: Optional[str]) -> Config:
    if path is None:
        return Config()
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    # simple recursive dataclass construction
    cfg = Config()
    for k, v in (raw or {}).items():
        if hasattr(cfg, k) and isinstance(getattr(cfg, k), (DataConfig, TokenizerConfig, OptimConfig, TrainConfig, GPTConfig, UTransformerConfig)):
            sub = getattr(cfg, k)
            for sk, sv in (v or {}).items():
                setattr(sub, sk, sv)
        else:
            setattr(cfg, k, v)
    return cfg


