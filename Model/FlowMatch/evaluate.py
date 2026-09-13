from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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
    spec = importlib.util.spec_from_file_location("flowmatch_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FlowMatch training module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["flowmatch_train"] = module
    spec.loader.exec_module(module)
    return module


advanced = load_training_module(THIS_DIR / "train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate advanced conditional FlowMatch residual model.")
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--solver", choices=["heun", "euler"], default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ssim-data-range", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.run_dir / "test_eval_advanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = core.select_device(args.device)
    ckpt_path = args.run_dir / f"{args.checkpoint}_model.pt"
    checkpoint = load_checkpoint(ckpt_path, device)
    ckpt_args = checkpoint.get("args", {})
    metadata = checkpoint["metadata"]
    stats = checkpoint.get("stats") or json.loads((args.run_dir / "normalization_stats.json").read_text())

    manifest, raw_metadata = core.build_manifest_from_index(args.test_index)
    if args.max_samples is not None and len(manifest) > args.max_samples:
        manifest = manifest.sample(args.max_samples, random_state=7).sort_index()
    source_x_channels = list(metadata.get("source_x_channels", raw_metadata["x_channels"]))
    x_indices = metadata.get("input_channel_indices")
    era_ssrd_idx = source_x_channels.index("era_ssrd")

    dataset = core.TaskBDataset(
        manifest,
        era_ssrd_idx,
        x_mean=np.asarray(stats["x_mean"], dtype=np.float32),
        x_std=np.asarray(stats["x_std"], dtype=np.float32),
        residual_mean=stats["residual_mean"],
        residual_std=stats["residual_std"],
        x_indices=x_indices,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_args = argparse.Namespace(
        base_channels=int(ckpt_args.get("base_channels", 64)),
        channel_multipliers=tuple(int(v) for v in ckpt_args.get("channel_multipliers", [1, 2, 4, 8])),
        res_blocks=int(ckpt_args.get("res_blocks", 2)),
    )
    model = advanced.make_model(model_args, len(metadata["x_channels"]), device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sample_steps = int(args.sample_steps if args.sample_steps is not None else ckpt_args.get("sample_steps", 50))
    solver = str(args.solver if args.solver is not None else ckpt_args.get("solver", "heun"))
    metrics = advanced.evaluate(
        model,
        loader,
        stats,
        sample_steps,
        solver,
        device,
        args.ssim_data_range,
        args.seed,
    )
    record = {
        "checkpoint": str(ckpt_path),
        "test_index": str(args.test_index),
        "samples": int(len(manifest)),
        "sample_steps": sample_steps,
        "solver": solver,
        **metrics,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    pd.DataFrame([record]).to_csv(out_dir / "test_metrics.csv", index=False)
    print(json.dumps(record, indent=2), flush=True)
    print(f"wrote {out_dir}", flush=True)
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
