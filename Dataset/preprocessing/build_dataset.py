from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"
MONTHLY_ERA_DIR = "ERA_tile_subregions_monthly_solar"
ACTIVE_ERA_VARS = ["ssrd", "ssr", "ssrdc", "fdir", "cdir"]


def load_dataset_builder():
    import dataset_builder

    return dataset_builder


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run preprocessing over monthly ERA tile-subregion GRIBs. "
            "Unknown args are passed through to the dataset builder."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tile-ids", type=int, nargs="+", default=list(range(1, 20)))
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--era-vars", nargs="+", default=list(ACTIVE_ERA_VARS))
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    return parser.parse_known_args()


def month_bounds(month: int) -> tuple[str, str]:
    if month < 1 or month > 12:
        raise ValueError(f"month must be 1..12, got {month}")
    start = f"2018-{month:02d}-01T00:00:00"
    if month == 12:
        end = "2019-01-01T00:00:00"
    else:
        end = f"2018-{month + 1:02d}-01T00:00:00"
    return start, end


def patch_cfgrib_group_open(builder: Any, era_vars: list[str]) -> None:
    original_open_dataset = builder.xr.open_dataset

    def open_dataset_group_aware(path: Path | str, *args: Any, **kwargs: Any):
        if kwargs.get("engine") != "cfgrib":
            return original_open_dataset(path, *args, **kwargs)

        backend_kwargs = dict(kwargs.get("backend_kwargs") or {})
        backend_kwargs.setdefault("indexpath", "")
        wanted = set(era_vars)
        attempts = [
            ("stepType=accum", {"stepType": "accum"}),
            ("stepType=avg", {"stepType": "avg"}),
            ("paramId radiation set", {"paramId": [169, 176, 177, 228021, 228022]}),
            ("no filter", None),
        ]
        errors: list[str] = []
        for label, filter_by_keys in attempts:
            attempt_kwargs = dict(backend_kwargs)
            if filter_by_keys is not None:
                attempt_kwargs["filter_by_keys"] = filter_by_keys
            print(f"Opening cfgrib group attempt: {label} for {path}", flush=True)
            try:
                ds = original_open_dataset(path, *args, **{**kwargs, "backend_kwargs": attempt_kwargs})
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                print(f"  failed: {errors[-1]}", flush=True)
                continue

            available = set(ds.data_vars)
            print(f"  opened vars: {sorted(available)}", flush=True)
            if wanted.issubset(available):
                print(f"Using cfgrib group: {label}", flush=True)
                return ds
            ds.close()
            errors.append(f"{label}: missing {sorted(wanted - available)}")

        raise KeyError(
            f"Could not open a cfgrib group in {path} containing {sorted(wanted)}. "
            f"Attempts: {' | '.join(errors)}"
        )

    builder.xr.open_dataset = open_dataset_group_aware


def monthly_grib_path(data_root: Path, tile_id: int, month: int) -> Path:
    return data_root / MONTHLY_ERA_DIR / f"ERA5_2018_tile_{tile_id:02d}_{month:02d}.grib"


def output_complete(run_out_dir: Path) -> bool:
    return (run_out_dir / "metadata.json").exists() and (run_out_dir / "samples_index.csv").exists()


def reused_output_dir(reuse_roots: list[Path], tile_id: int, month: int) -> Path | None:
    for root in reuse_roots:
        directory = root / f"tile_{tile_id:02d}" / f"month_{month:02d}"
        if output_complete(directory):
            return directory
    return None


def main() -> int:
    args, passthrough = parse_wrapper_args()
    builder = load_dataset_builder()
    patch_cfgrib_group_open(builder, args.era_vars)

    total_runs = len(args.tile_ids) * len(args.months)
    completed_runs = 0
    skipped_runs = 0
    original_argv = list(sys.argv)
    try:
        for tile_id in args.tile_ids:
            for month in args.months:
                grib_path = monthly_grib_path(args.data_root, tile_id, month)
                run_out_dir = args.out_dir / f"tile_{tile_id:02d}" / f"month_{month:02d}"

                if args.skip_existing and output_complete(run_out_dir):
                    skipped_runs += 1
                    print(f"Skipping existing output: tile={tile_id:02d} month={month:02d} out={run_out_dir}", flush=True)
                    continue
                if args.skip_existing:
                    reuse_dir = reused_output_dir(args.reuse_root, tile_id, month)
                    if reuse_dir is not None:
                        skipped_runs += 1
                        print(
                            f"Skipping reused output: tile={tile_id:02d} month={month:02d} "
                            f"reuse={reuse_dir}",
                            flush=True,
                        )
                        continue

                if not grib_path.exists():
                    message = f"Missing ERA monthly tile GRIB: {grib_path}"
                    if args.skip_missing:
                        skipped_runs += 1
                        print(f"Skipping: {message}", flush=True)
                        continue
                    raise FileNotFoundError(message)

                completed_runs += 1
                quarter_name = f"tile{tile_id:02d}_month{month:02d}"
                start, end = month_bounds(month)
                grib_name = str(Path("..") / MONTHLY_ERA_DIR / grib_path.name)

                print(
                    f"\nRun {completed_runs}/{total_runs}: tile={tile_id:02d} "
                    f"month={calendar.month_abbr[month]} grib={grib_path}",
                    flush=True,
                )

                builder.QUARTERS = {quarter_name: (grib_name, start, end)}
                builder.DEFAULT_OUT_DIR = run_out_dir

                sys.argv = [
                    original_argv[0],
                    "--data-root",
                    str(args.data_root),
                    "--tile-ids",
                    str(tile_id),
                    "--quarters",
                    quarter_name,
                    "--out-dir",
                    str(run_out_dir),
                    "--split",
                    args.split,
                    "--era-vars",
                    *args.era_vars,
                    *passthrough,
                ]
                result = int(builder.main())
                if result != 0:
                    return result
    finally:
        sys.argv = original_argv

    print(f"\nDone. Outputs under: {args.out_dir}")
    print(f"processed_runs={completed_runs} skipped_runs={skipped_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
