import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "train.py",
        "--model",
        "solarflowrefiner",
        "--",
        "--train-index",
        "splits/train_index.csv",
        "--val-index",
        "splits/val_index.csv",
        "--out-dir",
        "runs/SolarFlowRefiner",
    ]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
