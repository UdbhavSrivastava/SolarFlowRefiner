import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "runs" / "base_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        ("splits/train_index.csv", "runs/base_cache/train_base.npy"),
        ("splits/val_index.csv", "runs/base_cache/val_base.npy"),
        ("splits/test_index.csv", "runs/base_cache/test_base.npy"),
    ]
    for index_path, out_path in commands:
        status = subprocess.call(
            [
                sys.executable,
                "run_experiment.py",
                "--model",
                "flowrefiner-ode",
                "--stage",
                "precompute",
                "--",
                "--index",
                index_path,
                "--run-dir",
                "runs/FlowMatch",
                "--out",
                out_path,
            ],
            cwd=root,
        )
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
