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
    spec = importlib.util.spec_from_file_location("flowrefiner_pde_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FlowRefiner-PDE training module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["flowrefiner_pde_train"] = module
    spec.loader.exec_module(module)
    return module


pde_refiner = load_training_module(THIS_DIR / "train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FlowMatch-initialized PDE-Refiner.")
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--test-base-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
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
    refinement_steps = int(args.refinement_steps if args.refinement_steps is not None else ckpt_args.get("refinement_steps", 8))
    refine_strength = float(args.refine_strength if args.refine_strength is not None else ckpt_args.get("refine_strength", 1.0))

    ds, _meta, _channels = pde_refiner.build_dataset(
        args.test_index,
        stats,
        ",".join(metadata["x_channels"]),
        args.test_base_cache,
        args.max_samples,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    multipliers = ckpt_args.get("channel_multipliers", [1, 2, 4, 8])
    if isinstance(multipliers, str):
        multipliers = pde_refiner.parse_multipliers(multipliers)
    else:
        multipliers = tuple(int(v) for v in multipliers)
    model = pde_refiner.make_model(
        argparse.Namespace(
            base_channels=int(ckpt_args.get("base_channels", 64)),
            channel_multipliers=multipliers,
            res_blocks=int(ckpt_args.get("res_blocks", 2)),
        ),
        len(metadata["refiner_condition_channels"]),
        device,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = pde_refiner.evaluate(
        model,
        loader,
        stats,
        refinement_steps,
        refine_strength,
        device,
        args.ssim_data_range,
    )
    record = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "test_index": str(args.test_index),
        "test_base_cache": str(args.test_base_cache),
        "samples": int(len(ds)),
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
