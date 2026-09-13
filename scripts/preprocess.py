import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "preprocess.py", "check"],
        [
            sys.executable,
            "preprocess.py",
            "preprocess",
            "--tiles",
            "1-10,12,14",
            "--months",
            "1-12",
        ],
        [
            sys.executable,
            "preprocess.py",
            "splits",
            "--split-tiles",
            "1-10,12,14",
            "--split-months",
            "1-12",
            "--seed",
            "42",
        ],
    ]
    for command in commands:
        status = subprocess.call(command, cwd=root)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
