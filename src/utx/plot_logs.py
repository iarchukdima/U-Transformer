import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(csv_path: str):
    steps = []
    train_loss = []
    val_loss = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            tl = row.get("train_loss")
            vl = row.get("val_loss")
            train_loss.append(float(tl) if tl not in (None, "") else None)
            val_loss.append(float(vl) if vl not in (None, "") else None)
    return steps, train_loss, val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to training_log.csv")
    ap.add_argument("--out", type=str, default=None, help="Output image path (png)")
    args = ap.parse_args()

    steps, train_loss, val_loss = load_log(args.log)

    # Prepare series with only defined points
    t_steps = [s for s, v in zip(steps, train_loss) if v is not None]
    t_vals = [v for v in train_loss if v is not None]
    v_steps = [s for s, v in zip(steps, val_loss) if v is not None]
    v_vals = [v for v in val_loss if v is not None]

    plt.figure(figsize=(10, 6))
    if t_steps:
        plt.plot(t_steps, t_vals, label="train_loss", alpha=0.7)
    if v_steps:
        plt.plot(v_steps, v_vals, label="val_loss", alpha=0.7)
    plt.xlabel("global_step")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = args.out or str(Path(args.log).with_suffix(".png"))
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()


