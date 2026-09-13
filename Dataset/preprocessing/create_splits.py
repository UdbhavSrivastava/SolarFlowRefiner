from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "splits"


def parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def month_dir(root: Path, tile_id: int, month: int) -> Path:
    return root / f"tile_{tile_id:02d}" / f"month_{month:02d}"


def find_month_dir(roots: list[Path], tile_id: int, month: int) -> Path | None:
    for root in roots:
        directory = month_dir(root, tile_id, month)
        if (directory / "samples_index.csv").exists():
            return directory
    return None


def load_manifest(roots: list[Path], tile_ids: list[int], months: list[int], skip_missing: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for tile_id in tile_ids:
        for month in months:
            directory = find_month_dir(roots, tile_id, month)
            if directory is None:
                index_path = month_dir(roots[0], tile_id, month) / "samples_index.csv"
                missing.append(index_path)
                if skip_missing:
                    continue
                raise FileNotFoundError(index_path)
            index_path = directory / "samples_index.csv"
            if not index_path.exists():
                missing.append(index_path)
                if skip_missing:
                    continue
                raise FileNotFoundError(index_path)
            index = pd.read_csv(index_path)
            if "chunk" in index.columns:
                index["chunk"] = index["chunk"].astype(str).str.replace("\\", "/", regex=False)
            index["tile_id"] = tile_id
            index["month"] = month
            index["month_dir"] = str(directory)
            frames.append(index)
    if missing:
        print(f"missing_sample_indexes={len(missing)}")
        for path in missing[:20]:
            print(f"missing: {path}")
        if len(missing) > 20:
            print(f"... {len(missing) - 20} more")
    if not frames:
        raise RuntimeError("No sample indexes found")
    manifest = pd.concat(frames, ignore_index=True)
    manifest["utc_time"] = pd.to_datetime(manifest["utc_time"], utc=True)
    manifest["date"] = manifest["utc_time"].dt.date.astype(str)
    manifest["hour"] = manifest["utc_time"].dt.hour
    return manifest


def assign_day_blocks(group: pd.DataFrame, rng: np.random.Generator, train_frac: float, val_frac: float) -> pd.Series:
    day_counts = group.groupby("date", sort=True).size().reset_index(name="samples")
    shuffled = day_counts.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    total = int(shuffled["samples"].sum())
    train_target = total * train_frac
    val_target = total * val_frac
    counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": train_target, "val": val_target, "test": total - train_target - val_target}
    assignment: dict[str, str] = {}

    for row in shuffled.itertuples(index=False):
        deficits = {split: targets[split] - counts[split] for split in counts}
        split = max(deficits, key=deficits.get)
        assignment[str(row.date)] = split
        counts[split] += int(row.samples)

    return group["date"].map(assignment)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create tile/month stratified day-block train/validation/test splits.")
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tile-ids", default="1-19")
    parser.add_argument("--months", default="1-12")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {frac_sum}")

    tile_ids = parse_int_list(args.tile_ids)
    months = parse_int_list(args.months)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    roots = [args.preprocess_root, *args.reuse_root]
    manifest = load_manifest(roots, tile_ids, months, args.skip_missing)
    rng = np.random.default_rng(args.seed)
    split_values = []
    for (_tile, _month), group in manifest.groupby(["tile_id", "month"], sort=True):
        split_values.append(assign_day_blocks(group, rng, args.train_frac, args.val_frac))
    manifest["split_name"] = pd.concat(split_values).sort_index()

    keep_cols = [
        "chunk",
        "row",
        "split",
        "split_name",
        "quarter",
        "tile_id",
        "site_id",
        "site_name",
        "month",
        "date",
        "hour",
        "hour_index",
        "frame_start",
        "utc_time",
        "era_valid_time",
        "era_delta_minutes",
        "era_time_index",
        "era_step_index",
        "target_ssr_mean",
        "era_ssrd_mean",
        "era_ssr_mean",
        "cm_mean",
        "month_dir",
    ]
    keep_cols = [col for col in keep_cols if col in manifest.columns]

    for split in ["train", "val", "test"]:
        out_path = args.out_dir / f"{split}_index.csv"
        rows = manifest["split_name"] == split
        manifest.loc[rows, keep_cols].to_csv(out_path, index=False)
        print(f"{split}: {out_path} rows={int(rows.sum())}")

    summary = (
        manifest.groupby(["tile_id", "month", "split_name"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values(["tile_id", "month", "split_name"])
    )
    summary.to_csv(args.out_dir / "split_summary_by_tile_month.csv", index=False)

    hour_summary = (
        manifest.groupby(["split_name", "hour"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values(["split_name", "hour"])
    )
    hour_summary.to_csv(args.out_dir / "split_summary_by_hour.csv", index=False)

    overall = manifest.groupby("split_name").size().reset_index(name="samples")
    overall["fraction"] = overall["samples"] / overall["samples"].sum()
    overall.to_csv(args.out_dir / "split_summary_overall.csv", index=False)
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
