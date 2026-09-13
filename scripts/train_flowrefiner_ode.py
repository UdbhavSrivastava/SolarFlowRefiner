import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "train.py",
        "--model",
        "flowrefiner-ode",
        "--",
        "--train-index",
        "splits/train_index.csv",
        "--val-index",
        "splits/val_index.csv",
        "--train-base-cache",
        "runs/base_cache/train_base.npy",
        "--val-base-cache",
        "runs/base_cache/val_base.npy",
        "--base-run-dir",
        "runs/FlowMatch",
        "--out-dir",
        "runs/FlowRefiner-ODE",
    ]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
