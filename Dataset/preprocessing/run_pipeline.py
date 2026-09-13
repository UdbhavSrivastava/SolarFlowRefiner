from __future__ import annotations

import argparse
import tempfile
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "splits"
DEFAULT_REUSE_ROOT = None


def parse_range(value: str) -> list[str]:
    values: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(str(item) for item in range(int(start), int(end) + 1))
        else:
            values.append(str(int(part)))
    return values


def run_logged(name: str, command: list[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    temp_log_path = Path(tempfile.gettempdir()) / f"pipeline_{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    active_log_path = log_path
    try:
        with log_path.open("a", encoding="utf-8"):
            pass
    except PermissionError:
        active_log_path = temp_log_path
    print(f"\n== {name} ==")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print(f"log={active_log_path}")

    with active_log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    if active_log_path != log_path:
        print(f"Could not write to package log directory. Log kept at: {active_log_path}")

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def check(args: argparse.Namespace) -> None:
    run_logged(
        "check_inputs",
        [
            sys.executable,
            str(PREPROCESSING_DIR / "check_inputs.py"),
            "--root",
            str(args.data_root),
            "--era-dir",
            str(args.data_root / "ERA_tile_subregions_monthly_solar"),
            "--solarcube-dir",
            str(args.data_root / "Solarcube"),
            "--tiles",
            args.tiles,
            "--months",
            args.months,
        ],
    )


def preprocess(args: argparse.Namespace) -> None:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PREPROCESSING_DIR / "build_dataset.py"),
        "--data-root",
        str(args.data_root),
        "--tile-ids",
        *parse_range(args.tiles),
        "--months",
        *parse_range(args.months),
        "--out-dir",
        str(args.data_dir),
        "--solarcube-time-basis",
        "utc",
        "--era-accumulation-mode",
        "divide",
        "--era-match-policy",
        "primary_max",
        "--min-era-ssrd-mean",
        "10",
        "--min-valid-target-pct",
        str(args.min_valid_target_pct),
        "--min-valid-visible-pct",
        str(args.min_valid_visible_pct),
        "--chunk-size",
        str(args.chunk_size),
        "--era-read-retries",
        "3",
        "--era-retry-sleep",
        "2",
        "--progress-every",
        str(args.progress_every),
        "--save-era-coarse",
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.reuse_root is not None:
        command.extend(["--reuse-root", str(args.reuse_root)])
    run_logged("preprocess", command)


def splits(args: argparse.Namespace) -> None:
    args.split_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PREPROCESSING_DIR / "create_splits.py"),
        "--preprocess-root",
        str(args.data_dir),
        "--out-dir",
        str(args.split_dir),
        "--tile-ids",
        args.split_tiles,
        "--months",
        args.split_months,
        "--train-frac",
        "0.70",
        "--val-frac",
        "0.10",
        "--test-frac",
        "0.20",
        "--seed",
        str(args.seed),
    ]
    if args.reuse_root is not None:
        command.extend(["--reuse-root", str(args.reuse_root)])
    run_logged("create_splits", command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline: input check, dataset construction, and splits.")
    parser.add_argument("stage", choices=["check", "preprocess", "splits", "all"])
    parser.add_argument("--tiles", default="1-19")
    parser.add_argument("--months", default="1-12")
    parser.add_argument("--split-tiles", default="1-19")
    parser.add_argument("--split-months", default="1-12")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--reuse-root", type=Path, default=DEFAULT_REUSE_ROOT)
    parser.add_argument("--min-valid-target-pct", type=float, default=50.0)
    parser.add_argument("--min-valid-visible-pct", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    if args.stage in {"check", "all"}:
        check(args)
    if args.stage in {"preprocess", "all"}:
        preprocess(args)
    if args.stage in {"splits", "all"}:
        splits(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
