import argparse
import os
import csv
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .config import load_config, Config
from .data import make_dataloaders
from .utils import get_device, CosineLRScheduler, save_checkpoint
from .models.gpt_small import GPTSmall
from .models.u_transformer import UTransformer


def build_model(cfg: Config, vocab_size: int) -> nn.Module:
    if cfg.model_type == "gpt":
        m = GPTSmall(
            vocab_size=vocab_size,
            max_seq_len=cfg.data.max_seq_len,
            n_layers=cfg.gpt.n_layers,
            n_heads=cfg.gpt.n_heads,
            embed_dim=cfg.gpt.embed_dim,
            mlp_ratio=cfg.gpt.mlp_ratio,
            dropout=cfg.gpt.dropout,
            tie_weights=cfg.gpt.tie_weights,
        )
    else:
        m = UTransformer(
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
            tie_weights=cfg.u.tie_weights,
        )
    return m


def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
    model.train()
    return total_loss / max(1, count)


def train(cfg: Config):
    device = get_device(cfg.train.device)
    train_dl, val_dl, vocab_size = make_dataloaders(
        file=cfg.data.file,
        train_split=cfg.data.train_split,
        max_seq_len=cfg.data.max_seq_len,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.data.shuffle,
        tokenizer_cfg=cfg.tokenizer,
        data_cfg=cfg.data,
    )

    model = build_model(cfg, vocab_size).to(device)
    opt = AdamW(model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay)
    # support iterable datasets with unknown length
    iterable = False
    try:
        dl_len = len(train_dl)  # may raise TypeError for iterable datasets
        steps_per_epoch = dl_len // max(1, cfg.train.grad_accum_steps)
    except TypeError:
        iterable = True
        steps_per_epoch = cfg.train.steps_per_epoch or 1000
    max_steps = cfg.train.epochs * max(1, steps_per_epoch)
    sched = CosineLRScheduler(opt, max_steps=max_steps, warmup_steps=cfg.optim.warmup_steps, base_lr=cfg.optim.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.train.amp and device.type == "cuda"))

    global_step = 0
    best_val_loss = float("inf")
    bad_evals = 0
    patience = getattr(cfg.train, "early_stop_patience", 5)
    stop_training = False

    # prepare CSV logging
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.train.ckpt_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "epoch", "lr", "train_loss", "val_loss"]) 
    for epoch in range(cfg.train.epochs):
        pbar = tqdm(train_dl, desc=f"epoch {epoch}", total=steps_per_epoch if iterable else None)
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast('cuda', enabled=(cfg.train.amp and device.type == "cuda")):
                _, loss = model(x, y)
                loss = loss / cfg.train.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % cfg.train.grad_accum_steps == 0:
                # optional gradient clipping (AMP-aware)
                if cfg.train.grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                lr = sched.step()
                global_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
                # log train loss
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([global_step, epoch, lr, f"{loss.item():.8f}", ""])
                if global_step % cfg.train.eval_interval == 0:
                    # cap validation batches for streaming
                    model.eval()
                    total_loss = 0.0
                    vb = 0
                    with torch.no_grad():
                        for xv, yv in val_dl:
                            xv = xv.to(device)
                            yv = yv.to(device)
                            _, vloss = model(xv, yv)
                            total_loss += vloss.item()
                            vb += 1
                            if iterable and vb >= cfg.train.val_batches:
                                break
                    model.train()
                    val_loss = total_loss / max(1, vb)
                    pbar.write(f"val_loss={val_loss:.4f}")
                    # log val loss
                    with open(log_path, "a", newline="") as f:
                        csv.writer(f).writerow([global_step, epoch, lr, "", f"{val_loss:.8f}"])
                    # Early stopping and best checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        bad_evals = 0
                        os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
                        save_checkpoint(os.path.join(cfg.train.ckpt_dir, "best.pt"), {
                            "model": model.state_dict(),
                            "cfg": cfg,
                            "global_step": global_step,
                            "val_loss": val_loss,
                        })
                        pbar.write(f"best_val_loss={val_loss:.4f} (checkpoint saved)")
                    else:
                        bad_evals += 1
                        if bad_evals >= patience:
                            pbar.write(f"Early stopping: no improvement in {patience} evals")
                            stop_training = True
                if iterable and (step + 1) >= steps_per_epoch:
                    break
            if stop_training:
                break
        # ensure at least one validation per epoch
        try:
            val_loss_epoch = evaluate(model, val_dl, device)
            print(f"epoch_end_val_loss={val_loss_epoch:.4f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([global_step, epoch, lr, "", f"{val_loss_epoch:.8f}"])
        except Exception:
            pass
        # save epoch checkpoint
        os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
        save_checkpoint(os.path.join(cfg.train.ckpt_dir, f"{cfg.model_type}_epoch{epoch}.pt"), {
            "model": model.state_dict(),
            "cfg": cfg,
        })
        if stop_training:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data.file", dest="data_file", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_file:
        cfg.data.file = args.data_file
    train(cfg)


if __name__ == "__main__":
    main()


