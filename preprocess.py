import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SolarFlowRefiner dataset preprocessing."
    )
    parser.add_argument(
        "stage",
        choices=["check", "preprocess", "splits", "all"],
        help="Preprocessing stage to run.",
    )
    parser.add_argument(
        "stage_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the preprocessing pipeline.",
    )
    args = parser.parse_args()

    forwarded = args.stage_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    pipeline = Path(__file__).resolve().parent / "Dataset" / "preprocessing" / "run_pipeline.py"
    sys.argv = [str(pipeline), args.stage, *forwarded]
    runpy.run_path(str(pipeline), run_name="__main__")


if __name__ == "__main__":
    main()
