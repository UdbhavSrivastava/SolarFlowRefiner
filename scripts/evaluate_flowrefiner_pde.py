import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "evaluate.py",
        "--model",
        "flowrefiner-pde",
        "--",
        "--test-index",
        "splits/test_index.csv",
        "--test-base-cache",
        "runs/base_cache/test_base.npy",
        "--run-dir",
        "runs/FlowRefiner-PDE",
        "--out-dir",
        "runs/FlowRefiner-PDE/test_eval",
    ]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
