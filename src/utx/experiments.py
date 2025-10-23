import argparse
from copy import deepcopy

from .config import Config
from .train import train


def run_all(data_file: str | None):
    # Baseline
    cfg = Config()
    cfg.model_type = "gpt"
    if data_file:
        cfg.data.file = data_file
    cfg.train.epochs = 1
    train(cfg)

    # U-Transformer constant_heads
    cfg2 = deepcopy(cfg)
    cfg2.model_type = "u_transformer"
    cfg2.u.head_strategy = "constant_heads"
    cfg2.train.epochs = 1
    train(cfg2)

    # U-Transformer constant_head_dim
    cfg3 = deepcopy(cfg)
    cfg3.model_type = "u_transformer"
    cfg3.u.head_strategy = "constant_head_dim"
    cfg3.train.epochs = 1
    train(cfg3)

    # No skips ablation
    cfg4 = deepcopy(cfg2)
    cfg4.u.use_skips = False
    train(cfg4)

    # Factor 4 downsample
    cfg5 = deepcopy(cfg2)
    cfg5.u.downsample_factor = 4
    train(cfg5)

    # Deeper bottleneck
    cfg6 = deepcopy(cfg2)
    cfg6.u.stages = [384, 256, 192, 128]
    cfg6.u.blocks_per_stage = [2, 2, 2, 4]
    train(cfg6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data.file", dest="data_file", type=str, default=None)
    args = ap.parse_args()
    run_all(args.data_file)


if __name__ == "__main__":
    main()


