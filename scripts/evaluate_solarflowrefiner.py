import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "evaluate.py",
        "--model",
        "solarflowrefiner",
        "--",
        "--test-index",
        "splits/test_index.csv",
        "--run-dir",
        "runs/SolarFlowRefiner",
        "--out-dir",
        "runs/SolarFlowRefiner/test_eval",
    ]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
