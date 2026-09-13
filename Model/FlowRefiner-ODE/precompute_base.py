from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FLOWMATCH_DIR = THIS_DIR.parent / "FlowMatch"
for path in (FLOWMATCH_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import core
import train as flowmatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen FlowMatch base residuals for FlowMatch-Refiner.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--out", type=Path, required=True, help="Output .npy file for normalized FlowMatch residuals.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes:.0f}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f}h {minutes:.0f}m"


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = core.select_device(args.device)
    ckpt_path = args.run_dir / f"{args.checkpoint}_model.pt"
    checkpoint = load_checkpoint(ckpt_path, device)
    ckpt_args = checkpoint.get("args", {})
    metadata = checkpoint["metadata"]
    stats = checkpoint.get("stats") or json.loads((args.run_dir / "normalization_stats.json").read_text())

    manifest, raw_metadata = core.build_manifest_from_index(args.index)
    if args.max_samples is not None:
        manifest = manifest.iloc[: args.max_samples].copy()

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
    model = flowmatch.make_model(model_args, len(metadata["x_channels"]), device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    first = dataset[0]["residual"]
    shape = (len(dataset), int(first.shape[0]), int(first.shape[1]), int(first.shape[2]))
    dtype = np.float16 if args.dtype == "float16" else np.float32
    cache = np.lib.format.open_memmap(args.out, mode="w+", dtype=dtype, shape=shape)

    started = time.perf_counter()
    offset = 0
    print(f"device={device}", flush=True)
    print(f"samples={len(dataset)} out={args.out} shape={shape} dtype={args.dtype}", flush=True)
    print(f"flowmatch_steps={args.sample_steps} solver={args.solver}", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="precompute flowmatch base")):
            cond = batch["x"].to(device)
            residual_shape = tuple(batch["residual"].shape)
            base = flowmatch.flow_sample(
                model,
                cond,
                residual_shape,
                int(args.sample_steps),
                str(args.solver),
                int(args.seed) + batch_idx,
            )
            base_np = base.detach().cpu().numpy().astype(dtype, copy=False)
            n = base_np.shape[0]
            cache[offset : offset + n] = base_np
            offset += n
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(loader) - batch_idx - 1) / max(batch_idx + 1, 1)
            print(
                f"[{offset}/{len(dataset)}] cached batch={batch_idx + 1}/{len(loader)} "
                f"elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}",
                flush=True,
            )
    cache.flush()
    meta = {
        "index": str(args.index),
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "samples": int(len(dataset)),
        "shape": list(shape),
        "dtype": args.dtype,
        "seed": int(args.seed),
        "sample_steps": int(args.sample_steps),
        "solver": str(args.solver),
        "stats_source": str(args.run_dir / "normalization_stats.json"),
        "cache_values": "normalized FlowMatch residuals",
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest.to_csv(args.out.with_suffix(args.out.suffix + ".manifest.csv"), index=False)
    dataset.close()
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
