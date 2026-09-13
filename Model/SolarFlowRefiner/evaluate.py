from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import core


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("solarflowrefiner_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SolarFlowRefiner training module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["solarflowrefiner_train"] = module
    spec.loader.exec_module(module)
    return module


solarflowrefiner = load_training_module(THIS_DIR / "train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SolarFlowRefiner.")
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--flowmatch-sample-steps", type=int, default=None)
    parser.add_argument("--flowmatch-solver", choices=["heun", "euler"], default=None)
    parser.add_argument("--refinement-steps", type=int, default=None)
    parser.add_argument("--refine-strength", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ssim-data-range", type=float, default=1000.0)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.run_dir / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = core.select_device(args.device)
    ckpt_path = args.run_dir / f"{args.checkpoint}_model.pt"
    checkpoint = load_checkpoint(ckpt_path, device)
    ckpt_args = checkpoint.get("args", {})
    metadata = checkpoint["metadata"]
    stats = checkpoint.get("stats") or json.loads((args.run_dir / "normalization_stats.json").read_text())

    flowmatch_sample_steps = int(
        args.flowmatch_sample_steps
        if args.flowmatch_sample_steps is not None
        else ckpt_args.get("flowmatch_sample_steps", 8)
    )
    flowmatch_solver = str(
        args.flowmatch_solver
        if args.flowmatch_solver is not None
        else ckpt_args.get("flowmatch_solver", "euler")
    )
    refinement_steps = int(
        args.refinement_steps
        if args.refinement_steps is not None
        else ckpt_args.get("refinement_steps", 8)
    )
    refine_strength = float(
        args.refine_strength
        if args.refine_strength is not None
        else ckpt_args.get("refine_strength", 1.0)
    )

    manifest, raw_metadata = core.build_manifest_from_index(args.test_index)
    source_x_channels = list(metadata.get("source_x_channels", raw_metadata["x_channels"]))
    era_ssrd_idx = source_x_channels.index("era_ssrd")
    x_indices = metadata.get("input_channel_indices")
    ds = solarflowrefiner.build_dataset(
        manifest,
        era_ssrd_idx,
        stats,
        x_indices,
        args.max_samples,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    flow_model = solarflowrefiner.make_flow_model(
        argparse.Namespace(
            base_channels=int(ckpt_args.get("base_channels", 64)),
            channel_multipliers=tuple(int(v) for v in ckpt_args.get("channel_multipliers", [1, 2, 4, 8])),
            res_blocks=int(ckpt_args.get("res_blocks", 2)),
        ),
        len(metadata["x_channels"]),
        device,
    )
    refiner = solarflowrefiner.make_refiner(
        argparse.Namespace(
            refiner_base_channels=int(ckpt_args.get("refiner_base_channels", 64)),
            refiner_channel_multipliers=tuple(int(v) for v in ckpt_args.get("refiner_channel_multipliers", [1, 2, 4, 8])),
            refiner_res_blocks=int(ckpt_args.get("refiner_res_blocks", 2)),
        ),
        len(metadata["x_channels"]),
        device,
    )
    flow_model.load_state_dict(checkpoint["flow_model"])
    refiner.load_state_dict(checkpoint["refiner_model"])
    flow_model.eval()
    refiner.eval()

    eval_args = argparse.Namespace(
        flowmatch_sample_steps=flowmatch_sample_steps,
        flowmatch_solver=flowmatch_solver,
        refinement_steps=refinement_steps,
        refine_strength=refine_strength,
        ssim_data_range=args.ssim_data_range,
    )
    metrics = solarflowrefiner.evaluate(flow_model, refiner, loader, stats, eval_args, device)
    record = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_index": str(args.test_index),
        "samples": int(len(ds)),
        "flowmatch_sample_steps": flowmatch_sample_steps,
        "flowmatch_solver": flowmatch_solver,
        "refinement_steps": refinement_steps,
        "refine_strength": refine_strength,
        **metrics,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    pd.DataFrame([record]).to_csv(out_dir / "test_metrics.csv", index=False)
    print(json.dumps(record, indent=2), flush=True)
    print(f"wrote {out_dir}", flush=True)
    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
