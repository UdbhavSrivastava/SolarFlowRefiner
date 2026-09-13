from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    ("flowmatch", "train"): PROJECT_ROOT / "Model" / "FlowMatch" / "train.py",
    ("flowmatch", "evaluate"): PROJECT_ROOT / "Model" / "FlowMatch" / "evaluate.py",
    ("flowrefiner-ode", "precompute"): PROJECT_ROOT
    / "Model"
    / "FlowRefiner-ODE"
    / "precompute_base.py",
    ("flowrefiner-ode", "train"): PROJECT_ROOT / "Model" / "FlowRefiner-ODE" / "train.py",
    ("flowrefiner-ode", "evaluate"): PROJECT_ROOT / "Model" / "FlowRefiner-ODE" / "evaluate.py",
    ("flowrefiner-pde", "train"): PROJECT_ROOT / "Model" / "FlowRefiner-PDE" / "train.py",
    ("flowrefiner-pde", "evaluate"): PROJECT_ROOT / "Model" / "FlowRefiner-PDE" / "evaluate.py",
    ("solarflowrefiner", "train"): PROJECT_ROOT
    / "Model"
    / "SolarFlowRefiner"
    / "train.py",
    ("solarflowrefiner", "evaluate"): PROJECT_ROOT
    / "Model"
    / "SolarFlowRefiner"
    / "evaluate.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one SolarFlowRefiner paper model with manually selected model/stage."
    )
    parser.add_argument("--model", choices=["flowmatch", "flowrefiner-ode", "flowrefiner-pde", "solarflowrefiner"], required=True)
    parser.add_argument("--stage", choices=["precompute", "train", "evaluate"], required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the selected script.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = SCRIPTS.get((args.model, args.stage))
    if script is None:
        valid = sorted({stage for model, stage in SCRIPTS if model == args.model})
        raise SystemExit(f"Stage {args.stage!r} is not available for {args.model}. Valid stages: {', '.join(valid)}")
    forwarded = list(args.script_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return subprocess.call([sys.executable, str(script), *forwarded], cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
