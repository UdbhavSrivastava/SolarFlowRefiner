from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FLOWMATCH_DIR = THIS_DIR.parent / "FlowMatch"
PDE_DIR = THIS_DIR.parent / "FlowRefiner-PDE"
for path in (FLOWMATCH_DIR, PDE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import core


def load_training_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


flowmatch = load_training_module("flowmatch_train", FLOWMATCH_DIR / "train.py")
pde = load_training_module("flowrefiner_pde_train", PDE_DIR / "train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FlowMatch generator and PDE-style refiner jointly from random initialization."
    )
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--input-channels", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--flowmatch-lr", type=float, default=2e-6)
    parser.add_argument("--refiner-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--flowmatch-loss-weight", type=float, default=1.0)
    parser.add_argument("--refiner-loss-weight", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-multipliers", type=flowmatch.parse_multipliers, default=flowmatch.parse_multipliers("1,2,4,8"))
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--flowmatch-sample-steps", type=int, default=8)
    parser.add_argument("--flowmatch-solver", choices=["heun", "euler"], default="euler")
    parser.add_argument("--time-sampling", choices=["logit_normal", "uniform"], default="logit_normal")
    parser.add_argument("--flowmatch-loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--refiner-base-channels", type=int, default=64)
    parser.add_argument("--refiner-channel-multipliers", type=pde.parse_multipliers, default=pde.parse_multipliers("1,2,4,8"))
    parser.add_argument("--refiner-res-blocks", type=int, default=2)
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument("--sigma-max", type=float, default=0.35)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--refine-strength", type=float, default=1.0)
    parser.add_argument("--train-state-mode", choices=["base_to_target", "target_noise"], default="base_to_target")
    parser.add_argument("--refiner-loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--mse-loss-weight", type=float, default=0.1)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.05)
    parser.add_argument("--base-consistency-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--ssim-data-range", type=float, default=1000.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def sample_times(batch_size: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand((batch_size, 1, 1, 1), device=device)
    else:
        t = torch.sigmoid(torch.randn((batch_size, 1, 1, 1), device=device))
    return t.clamp(1e-4, 1.0 - 1e-4)


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_name: str) -> torch.Tensor:
    values = torch.square(pred - target) if loss_name == "mse" else torch.abs(pred - target)
    return values.mul(mask).sum() / mask.sum().clamp_min(1.0)


def build_dataset(
    manifest,
    era_ssrd_idx: int,
    stats: dict,
    x_indices: list[int] | None,
    max_samples: int | None,
):
    return core.TaskBDataset(
        manifest,
        era_ssrd_idx,
        x_mean=np.asarray(stats["x_mean"], dtype=np.float32),
        x_std=np.asarray(stats["x_std"], dtype=np.float32),
        residual_mean=stats["residual_mean"],
        residual_std=stats["residual_std"],
        x_indices=x_indices,
        max_samples=max_samples,
    )


def make_flow_model(args: argparse.Namespace, cond_channels: int, device: torch.device) -> nn.Module:
    return flowmatch.make_model(
        argparse.Namespace(
            base_channels=int(args.base_channels),
            channel_multipliers=tuple(int(v) for v in args.channel_multipliers),
            res_blocks=int(args.res_blocks),
        ),
        cond_channels,
        device,
    )


def make_refiner(args: argparse.Namespace, cond_channels: int, device: torch.device) -> nn.Module:
    return pde.make_model(
        argparse.Namespace(
            base_channels=int(args.refiner_base_channels),
            channel_multipliers=tuple(int(v) for v in args.refiner_channel_multipliers),
            res_blocks=int(args.refiner_res_blocks),
        ),
        cond_channels + 1,
        device,
    )


def differentiable_flow_sample(
    model: nn.Module,
    cond: torch.Tensor,
    residual_shape: tuple[int, int, int, int],
    steps: int,
    solver: str,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    device = cond.device
    z = torch.randn(residual_shape, device=device) if noise is None else noise
    n_steps = max(int(steps), 1)
    dt = 1.0 / float(n_steps)
    bsz = int(residual_shape[0])
    for step in range(n_steps):
        t0_float = step / float(n_steps)
        t1_float = (step + 1) / float(n_steps)
        t0 = torch.full((bsz,), t0_float * 999.0, device=device)
        v0 = model(z, cond, t0)
        if solver == "heun" and step < n_steps - 1:
            z_euler = z + dt * v0
            t1 = torch.full((bsz,), t1_float * 999.0, device=device)
            v1 = model(z_euler, cond, t1)
            z = z + 0.5 * dt * (v0 + v1)
        else:
            z = z + dt * v0
    return z


def pde_refine_inference(
    model: nn.Module,
    cond: torch.Tensor,
    base_residual: torch.Tensor,
    steps: int,
    refine_strength: float,
) -> torch.Tensor:
    z = base_residual
    steps = max(int(steps), 1)
    strength = float(refine_strength)
    bsz = z.shape[0]
    for k_idx in range(steps):
        k = torch.full((bsz,), k_idx, device=z.device, dtype=torch.long)
        pred_clean = model(z, cond, pde.time_labels(k, steps))
        z = z + strength * (pred_clean - z)
    return z


@torch.no_grad()
def evaluate(flow_model, refiner, loader, stats, args, device):
    flow_model.eval()
    refiner.eval()
    residual_mean = float(stats["residual_mean"])
    residual_std = float(stats["residual_std"])
    abs_sum = sq_sum = base_abs_sum = base_sq_sum = 0.0
    count = 0
    target_means: list[float] = []
    pred_means: list[float] = []
    base_means: list[float] = []
    ssim_values: list[float] = []
    base_ssim_values: list[float] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x"].to(device)
        residual = batch["residual"].to(device)
        target = batch["target"].to(device)
        era = batch["era_ssrd"].to(device)
        mask = batch["mask"].to(device)
        base = differentiable_flow_sample(
            flow_model,
            x,
            tuple(residual.shape),
            args.flowmatch_sample_steps,
            args.flowmatch_solver,
        )
        cond = pde.make_cond(x, base)
        refined = pde_refine_inference(refiner, cond, base, args.refinement_steps, args.refine_strength)
        pred = era + refined * residual_std + residual_mean
        base_pred = era + base * residual_std + residual_mean
        diff = (pred - target) * mask
        base_diff = (base_pred - target) * mask
        abs_sum += float(diff.abs().sum().item())
        sq_sum += float(torch.square(diff).sum().item())
        base_abs_sum += float(base_diff.abs().sum().item())
        base_sq_sum += float(torch.square(base_diff).sum().item())
        count += int(mask.sum().item())
        for i in range(target.shape[0]):
            sample_mask = mask[i].bool()
            if sample_mask.any():
                target_means.append(float(target[i][sample_mask].mean().item()))
                pred_means.append(float(pred[i][sample_mask].mean().item()))
                base_means.append(float(base_pred[i][sample_mask].mean().item()))
                ssim_values.append(core.global_ssim(target[i, 0].detach().cpu().numpy(), pred[i, 0].detach().cpu().numpy(), args.ssim_data_range))
                base_ssim_values.append(core.global_ssim(target[i, 0].detach().cpu().numpy(), base_pred[i, 0].detach().cpu().numpy(), args.ssim_data_range))
    return {
        "mae": abs_sum / max(count, 1),
        "rmse": math.sqrt(sq_sum / max(count, 1)),
        "ssim": float(np.mean(ssim_values)) if ssim_values else np.nan,
        "base_mae": base_abs_sum / max(count, 1),
        "base_rmse": math.sqrt(base_sq_sum / max(count, 1)),
        "base_ssim": float(np.mean(base_ssim_values)) if base_ssim_values else np.nan,
        "sample_mean_corr": float(pd.Series(target_means).corr(pd.Series(pred_means))) if len(target_means) > 1 else np.nan,
        "base_sample_mean_corr": float(pd.Series(target_means).corr(pd.Series(base_means))) if len(target_means) > 1 else np.nan,
    }


def save_checkpoint(path: Path, flow_model, ema_flow, refiner, ema_refiner, optimizer, args, metadata, stats, epoch, metrics=None):
    payload = {
        "flow_model": ema_flow.state_dict(),
        "flow_raw_model": flow_model.state_dict(),
        "refiner_model": ema_refiner.state_dict(),
        "refiner_raw_model": refiner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metadata": metadata,
        "stats": stats,
        "epoch": epoch,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = core.select_device(args.device)

    train_manifest, raw_metadata = core.build_manifest_from_index(args.train_index)
    val_manifest, _ = core.build_manifest_from_index(args.val_index)
    source_x_channels = list(raw_metadata["x_channels"])
    era_ssrd_idx = source_x_channels.index("era_ssrd")
    x_channels, x_indices, metadata = core.select_input_channels(raw_metadata, args.input_channels)
    stats_path = args.stats_path if args.stats_path is not None else args.out_dir / "normalization_stats.json"
    stats = core.load_or_compute_stats(
        train_manifest,
        era_ssrd_idx,
        stats_path,
        mirror_path=args.out_dir / "normalization_stats.json",
        x_indices=x_indices,
    )

    train_ds = build_dataset(train_manifest, era_ssrd_idx, stats, x_indices, args.max_train_samples)
    val_ds = build_dataset(val_manifest, era_ssrd_idx, stats, x_indices, args.max_val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    flow_model = make_flow_model(args, len(x_channels), device)
    refiner = make_refiner(args, len(x_channels), device)
    ema_flow = copy.deepcopy(flow_model).to(device)
    ema_refiner = copy.deepcopy(refiner).to(device)
    ema_flow.eval()
    ema_refiner.eval()
    optimizer = torch.optim.AdamW(
        [
            {"params": flow_model.parameters(), "lr": args.flowmatch_lr},
            {"params": refiner.parameters(), "lr": args.refiner_lr},
        ],
        weight_decay=args.weight_decay,
    )
    sigmas = pde.sigma_schedule(args.refinement_steps, args.sigma_max, args.sigma_min, device)

    metadata = dict(metadata)
    metadata["source_x_channels"] = source_x_channels
    metadata["input_channel_indices"] = x_indices
    metadata["refiner_condition_channels"] = list(x_channels) + ["flowmatch_base_residual_norm"]
    run_config = {
        "method": "joint_flowmatch_pde_refiner_from_scratch",
        "args": json_safe(vars(args)),
        "metadata": metadata,
        "sigma_schedule": [float(v) for v in sigmas.detach().cpu().tolist()],
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "notes": [
            "FlowMatch and refiner are randomly initialized.",
            "No pretrained FlowMatch checkpoint is loaded.",
            "Each batch optimizes a FlowMatch velocity loss and a refinement reconstruction loss.",
            "The refinement loss also backpropagates through the differentiable FlowMatch sampler.",
        ],
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.out_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    history: list[dict[str, object]] = []
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    resume_path = args.resume_checkpoint or (args.out_dir / "last_model.pt" if args.resume else None)
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        flow_model.load_state_dict(checkpoint["flow_raw_model"])
        refiner.load_state_dict(checkpoint["refiner_raw_model"])
        ema_flow.load_state_dict(checkpoint["flow_model"])
        ema_refiner.load_state_dict(checkpoint["refiner_model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history_path = args.out_dir / "training_history.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            history_df = history_df[history_df["epoch"] < start_epoch]
            history = history_df.to_dict("records")
            val_mae = pd.to_numeric(history_df.get("val_mae", pd.Series(dtype=float)), errors="coerce")
            valid = history_df[val_mae.notna()]
            if not valid.empty:
                best_idx = val_mae.loc[valid.index].idxmin()
                best_mae = float(history_df.loc[best_idx, "val_mae"])
                best_epoch = int(history_df.loc[best_idx, "epoch"])
        print(f"resuming from {resume_path}; next_epoch={start_epoch}", flush=True)

    print(f"device={device}", flush=True)
    print("pretrained_flowmatch=False", flush=True)
    print(f"flowmatch_steps={args.flowmatch_sample_steps} solver={args.flowmatch_solver}", flush=True)
    print(f"input_channels={x_channels}", flush=True)
    print(f"condition_channels={metadata['refiner_condition_channels']}", flush=True)

    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        flow_model.train()
        refiner.train()
        total_losses: list[float] = []
        flow_losses: list[float] = []
        refine_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = batch["x"].to(device)
            residual = batch["residual"].to(device)
            mask = batch["mask"].to(device)
            bsz = residual.shape[0]

            t = sample_times(bsz, device, args.time_sampling)
            noise = torch.randn_like(residual)
            x_t = (1.0 - t) * noise + t * residual
            target_velocity = residual - noise
            pred_velocity = flow_model(x_t, x, t[:, 0, 0, 0] * 999.0)
            flow_loss = masked_loss(pred_velocity, target_velocity, mask, args.flowmatch_loss)

            base = differentiable_flow_sample(
                flow_model,
                x,
                tuple(residual.shape),
                args.flowmatch_sample_steps,
                args.flowmatch_solver,
            )
            cond = pde.make_cond(x, base)
            k = torch.randint(0, args.refinement_steps, (bsz,), device=device)
            sigma = sigmas[k].view(bsz, 1, 1, 1)
            noisy = pde.make_training_state(
                residual,
                base,
                sigma,
                k,
                args.refinement_steps,
                args.train_state_mode,
                torch.randn_like(residual),
            )
            pred_clean = refiner(noisy, cond, pde.time_labels(k, args.refinement_steps))
            primary = pde.masked_loss(pred_clean, residual, mask, args.refiner_loss)
            mse = pde.masked_loss(pred_clean, residual, mask, "mse")
            grad = pde.gradient_l1(pred_clean, residual, mask)
            refiner_loss = primary + args.mse_loss_weight * mse + args.gradient_loss_weight * grad
            if args.base_consistency_weight > 0:
                base_improved = torch.minimum(torch.abs(pred_clean - residual), torch.abs(base - residual))
                consistency = torch.relu(torch.abs(pred_clean - residual) - base_improved).mul(mask).sum() / mask.sum().clamp_min(1.0)
                refiner_loss = refiner_loss + args.base_consistency_weight * consistency

            loss = args.flowmatch_loss_weight * flow_loss + args.refiner_loss_weight * refiner_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
            nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            pde.unet.update_ema(ema_flow, flow_model, args.ema_decay)
            pde.unet.update_ema(ema_refiner, refiner, args.ema_decay)
            total_losses.append(float(loss.item()))
            flow_losses.append(float(flow_loss.item()))
            refine_losses.append(float(refiner_loss.item()))

        record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": float(np.mean(total_losses)),
            "train_flowmatch_loss": float(np.mean(flow_losses)),
            "train_refiner_loss": float(np.mean(refine_losses)),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(ema_flow, ema_refiner, val_loader, stats, args, device)
            record.update({f"val_{key}": value for key, value in metrics.items()})
            improved = metrics["mae"] < (best_mae - args.early_stopping_min_delta)
            if improved:
                best_mae = metrics["mae"]
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(args.out_dir / "best_model.pt", flow_model, ema_flow, refiner, ema_refiner, optimizer, args, metadata, stats, epoch, metrics)
            else:
                epochs_without_improvement += 1
            record["best_val_mae"] = best_mae
            record["best_epoch"] = best_epoch
            record["epochs_without_improvement"] = epochs_without_improvement
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                record["early_stopped"] = True
                stopped_early = True

        record["epoch_time_sec"] = float(time.time() - epoch_start)
        history.append(record)
        pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
        with (args.out_dir / "training_log.jsonl").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(json_safe(record)) + "\n")
        save_checkpoint(args.out_dir / "last_model.pt", flow_model, ema_flow, refiner, ema_refiner, optimizer, args, metadata, stats, epoch)
        print(record, flush=True)
        if stopped_early:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch} best_val_mae={best_mae:.6f}", flush=True)
            break

    train_ds.close()
    val_ds.close()
    print(f"wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
