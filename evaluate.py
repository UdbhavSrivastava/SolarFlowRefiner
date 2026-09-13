import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one of the SolarFlowRefiner paper models."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "flowmatch",
            "flowrefiner-ode",
            "flowrefiner-pde",
            "solarflowrefiner",
        ],
        help="Paper model to evaluate.",
    )
    parser.add_argument(
        "model_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected model evaluation script.",
    )
    args = parser.parse_args()

    forwarded = args.model_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    runner = Path(__file__).resolve().parent / "run_experiment.py"
    sys.argv = [
        str(runner),
        "--model",
        args.model,
        "--stage",
        "evaluate",
        "--",
        *forwarded,
    ]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
