from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_ERA_DIR = DEFAULT_ROOT / "ERA_tile_subregions_monthly_solar"
DEFAULT_SOLARCUBE_DIR = DEFAULT_ROOT / "Solarcube"


def parse_range(value: str) -> list[int]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check preprocessing input files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--era-dir", type=Path, default=DEFAULT_ERA_DIR)
    parser.add_argument("--solarcube-dir", type=Path, default=DEFAULT_SOLARCUBE_DIR)
    parser.add_argument("--tiles", default="1-19")
    parser.add_argument("--months", default="1-12")
    args = parser.parse_args()

    tiles = parse_range(args.tiles)
    months = parse_range(args.months)
    expected = {(tile, month) for tile in tiles for month in months}
    missing: list[Path] = []
    present: list[Path] = []

    for tile, month in sorted(expected):
        path = args.era_dir / f"ERA5_2018_tile_{tile:02d}_{month:02d}.grib"
        if path.exists():
            present.append(path)
        else:
            missing.append(path)

    print(f"root={args.root}")
    print(f"era_dir={args.era_dir}")
    print(f"solarcube_dir={args.solarcube_dir}")
    print(f"tiles={tiles[0]}-{tiles[-1]} count={len(tiles)}")
    print(f"months={months[0]}-{months[-1]} count={len(months)}")
    print(f"expected_gribs={len(expected)} present={len(present)} missing={len(missing)}")

    if present:
        total_gb = sum(path.stat().st_size for path in present) / (1024**3)
        print(f"present_grib_total_gb={total_gb:.3f}")

    if not args.solarcube_dir.exists():
        print(f"ERROR: missing SolarCube folder: {args.solarcube_dir}")
        return 2

    if missing:
        print("Missing GRIB files:")
        for path in missing:
            print(path)
        return 1

    print("Input check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
