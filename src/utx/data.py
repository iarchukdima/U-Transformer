from dataclasses import dataclass
from typing import Tuple, Optional
import os
import math

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from .tokenizer import ByteTokenizer, VOCAB_SIZE
from .config import TokenizerConfig, DataConfig


@dataclass
class TextDataSplits:
    train: torch.Tensor
    val: torch.Tensor


def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_splits(text: str, train_split: float, tokenizer: ByteTokenizer) -> TextDataSplits:
    # Document-aware split at paragraph-level, then pack with EOS separators
    import random
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    rnd = random.Random(1337)
    rnd.shuffle(paragraphs)
    num_train = int(len(paragraphs) * train_split)
    train_pars = paragraphs[:num_train]
    val_pars = paragraphs[num_train:]
    train_ids: list[int] = []
    for p in train_pars:
        train_ids.extend(tokenizer.encode(p))
    val_ids: list[int] = []
    for p in val_pars:
        val_ids.extend(tokenizer.encode(p))
    return TextDataSplits(train=torch.tensor(train_ids, dtype=torch.long), val=torch.tensor(val_ids, dtype=torch.long))


class TokenSeqDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, max_seq_len: int):
        self.tokens = tokens
        self.max_seq_len = max_seq_len

    def __len__(self):
        # -1 to ensure next token exists for label
        return max(0, len(self.tokens) - 1 - self.max_seq_len)

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.max_seq_len]
        y = self.tokens[idx + 1 : idx + 1 + self.max_seq_len]
        return x, y


def _encode_with_hf(text: str, cfg: TokenizerConfig, max_seq_len: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.hf_name, cache_dir=cfg.cache_dir)
    if cfg.add_special_tokens and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, add_special_tokens=cfg.add_special_tokens, return_attention_mask=False, return_tensors=None)
    ids = enc["input_ids"]
    if isinstance(ids[0], list):
        # if it produced multiple sequences, flatten
        ids = [i for seq in ids for i in seq]
    ids = torch.tensor(ids, dtype=torch.long)
    return ids, tok.vocab_size


class WikiStreamDataset(IterableDataset):
    def __init__(self, data_cfg: DataConfig, tok_cfg: TokenizerConfig, split: str, max_seq_len: int):
        super().__init__()
        self.data_cfg = data_cfg
        self.tok_cfg = tok_cfg
        self.split = split  # "train" or "validation" (Wikipedia doesn't have standard splits; we shard)
        self.max_seq_len = max_seq_len

    def _hf_tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.tok_cfg.hf_name, cache_dir=self.tok_cfg.cache_dir)
        if self.tok_cfg.add_special_tokens and tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _byte_tokenizer(self):
        return ByteTokenizer()

    def _dataset(self):
        from datasets import load_dataset
        ds = load_dataset("wikipedia", self.data_cfg.wiki_config, split="train", streaming=self.data_cfg.streaming)
        return ds

    def __iter__(self):
        ds = self._dataset()
        # simple split by hash modulo to simulate train/val
        def is_val(i):
            return (i % 100) < int((1 - self.data_cfg.train_split) * 100)

        if self.tok_cfg.type == "hf":
            tok = self._hf_tokenizer()
            def gen():
                buf = []
                for i, ex in enumerate(ds):
                    if (self.split == "validation" and not is_val(i)) or (self.split == "train" and is_val(i)):
                        continue
                    text = ex.get("text", "")
                    if not text:
                        continue
                    enc = tok(text, add_special_tokens=self.tok_cfg.add_special_tokens, return_attention_mask=False, truncation=False)
                    ids = enc["input_ids"]
                    for j in range(0, len(ids) - 1 - self.max_seq_len):
                        x = torch.tensor(ids[j:j + self.max_seq_len], dtype=torch.long)
                        y = torch.tensor(ids[j + 1:j + 1 + self.max_seq_len], dtype=torch.long)
                        yield x, y
            return gen()
        else:
            btok = self._byte_tokenizer()
            def gen():
                for i, ex in enumerate(ds):
                    if (self.split == "validation" and not is_val(i)) or (self.split == "train" and is_val(i)):
                        continue
                    text = ex.get("text", "")
                    if not text:
                        continue
                    ids = btok.encode(text)
                    for j in range(0, len(ids) - 1 - self.max_seq_len):
                        x = torch.tensor(ids[j:j + self.max_seq_len], dtype=torch.long)
                        y = torch.tensor(ids[j + 1:j + 1 + self.max_seq_len], dtype=torch.long)
                        yield x, y
            return gen()


def make_dataloaders(
    file: str,
    train_split: float,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool,
    tokenizer_cfg: Optional[TokenizerConfig] = None,
    data_cfg: Optional[DataConfig] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    data_cfg = data_cfg or DataConfig()
    if data_cfg.source == "hf_wikipedia":
        train_ds = WikiStreamDataset(data_cfg, tokenizer_cfg or TokenizerConfig(), split="train", max_seq_len=max_seq_len)
        val_ds = WikiStreamDataset(data_cfg, tokenizer_cfg or TokenizerConfig(), split="validation", max_seq_len=max_seq_len)
        # vocab size depends on tokenizer
        if tokenizer_cfg and tokenizer_cfg.type == "hf":
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_cfg.hf_name, cache_dir=tokenizer_cfg.cache_dir)
            vocab_size = tok.vocab_size
        else:
            vocab_size = VOCAB_SIZE
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        val_dl = DataLoader(val_ds, batch_size=batch_size)
        return train_dl, val_dl, vocab_size
    else:
        text = load_text_file(file)
        if tokenizer_cfg and tokenizer_cfg.type == "hf":
            ids, vocab_size = _encode_with_hf(text, tokenizer_cfg, max_seq_len)
            split = int(len(ids) * train_split)
            splits = TextDataSplits(train=ids[:split], val=ids[split:])
        else:
            tokenizer = ByteTokenizer()
            splits = build_splits(text, train_split, tokenizer)
            vocab_size = VOCAB_SIZE
        train_ds = TokenSeqDataset(splits.train, max_seq_len)
        val_ds = TokenSeqDataset(splits.val, max_seq_len)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_dl, val_dl, vocab_size


