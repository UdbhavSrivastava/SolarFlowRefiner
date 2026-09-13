from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils import core
from utils import unet


def parse_multipliers(value: str) -> tuple[int, ...]:
    return unet.parse_multipliers(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FlowMatch -> FlowRefiner residual model.")
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--train-base-cache", type=Path, required=True, help="Cached normalized FlowMatch residuals for train split.")
    parser.add_argument("--val-base-cache", type=Path, required=True, help="Cached normalized FlowMatch residuals for val split.")
    parser.add_argument("--base-run-dir", type=Path, required=True, help="FlowMatch base run directory.")
    parser.add_argument("--base-checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-channels", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-multipliers", type=parse_multipliers, default=parse_multipliers("1,2,4,8"))
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--time-sampling", choices=["logit_normal", "uniform"], default="logit_normal")
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.5)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.05)
    parser.add_argument("--train-noise-scale", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--eval-every", type=int, default=1)
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


class CachedBaseDataset(Dataset):
    def __init__(self, base: core.TaskBDataset, cache_path: Path, max_samples: int | None = None) -> None:
        self.base = base
        self.cache = np.load(cache_path, mmap_mode="r")
        self.length = len(base) if max_samples is None else min(int(max_samples), len(base))
        if len(self.cache) < self.length:
            raise ValueError(f"Cache has {len(self.cache)} rows but dataset uses {self.length} rows: {cache_path}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]
        item["base_residual"] = torch.from_numpy(np.asarray(self.cache[idx], dtype=np.float32).copy())
        return item

    def close(self) -> None:
        self.base.close()


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_name: str) -> torch.Tensor:
    values = torch.square(pred - target) if loss_name == "mse" else torch.abs(pred - target)
    return values.mul(mask).sum() / mask.sum().clamp_min(1.0)


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    mask_dx = mask[..., :, 1:] * mask[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_dy = mask[..., 1:, :] * mask[..., :-1, :]
    loss_x = torch.abs(pred_dx - target_dx).mul(mask_dx).sum()
    loss_y = torch.abs(pred_dy - target_dy).mul(mask_dy).sum()
    count = mask_dx.sum() + mask_dy.sum()
    return (loss_x + loss_y) / count.clamp_min(1.0)


def sample_times(batch_size: int, device: torch.device, mode: str, generator: torch.Generator | None = None) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand((batch_size, 1, 1, 1), device=device, generator=generator)
    else:
        t = torch.sigmoid(torch.randn((batch_size, 1, 1, 1), device=device, generator=generator))
    return t.clamp(1e-4, 1.0 - 1e-4)


def make_cond(x: torch.Tensor, base_residual: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, base_residual], dim=1)


def make_model(args: argparse.Namespace, cond_channels: int, device: torch.device) -> nn.Module:
    return unet.UNet(
        cond_channels=cond_channels,
        base_channels=int(args.base_channels),
        channel_multipliers=tuple(int(v) for v in args.channel_multipliers),
        res_blocks=int(args.res_blocks),
    ).to(device)


@torch.no_grad()
def refiner_sample(
    model: nn.Module,
    cond: torch.Tensor,
    base_residual: torch.Tensor,
    steps: int,
    solver: str,
) -> torch.Tensor:
    model.eval()
    z = base_residual.clone()
    n_steps = max(int(steps), 1)
    dt = 1.0 / float(n_steps)
    bsz = z.shape[0]
    for step in range(n_steps):
        t0_float = step / float(n_steps)
        t1_float = (step + 1) / float(n_steps)
        t0 = torch.full((bsz,), t0_float * 999.0, device=z.device)
        v0 = model(z, cond, t0)
        if solver == "heun" and step < n_steps - 1:
            z_euler = z + dt * v0
            t1 = torch.full((bsz,), t1_float * 999.0, device=z.device)
            v1 = model(z_euler, cond, t1)
            z = z + 0.5 * dt * (v0 + v1)
        else:
            z = z + dt * v0
    return z


@torch.no_grad()
def evaluate_flow_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    time_sampling: str,
    loss_name: str,
    train_noise_scale: float,
) -> float:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    loss_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="val_loss", leave=False):
        x = batch["x"].to(device)
        residual = batch["residual"].to(device)
        base = batch["base_residual"].to(device)
        mask = batch["mask"].to(device)
        cond = make_cond(x, base)
        bsz = residual.shape[0]
        t = sample_times(bsz, device, time_sampling, generator)
        interp = (1.0 - t) * base + t * residual
        if train_noise_scale > 0:
            sigma = float(train_noise_scale) * torch.sin(math.pi * t).clamp_min(0.0)
            interp = interp + sigma * torch.randn(residual.shape, device=device, generator=generator)
        target_velocity = residual - base
        pred_velocity = model(interp, cond, t[:, 0, 0, 0] * 999.0)
        values = torch.square(pred_velocity - target_velocity) if loss_name == "mse" else torch.abs(pred_velocity - target_velocity)
        loss_sum += float(values.mul(mask).sum().item())
        count += int(mask.sum().item())
    return loss_sum / max(count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    stats: dict,
    sample_steps: int,
    solver: str,
    device: torch.device,
    ssim_data_range: float,
) -> dict[str, float]:
    abs_sum = sq_sum = base_abs_sum = base_sq_sum = 0.0
    count = 0
    target_means: list[float] = []
    pred_means: list[float] = []
    base_means: list[float] = []
    ssim_values: list[float] = []
    base_ssim_values: list[float] = []
    residual_mean = float(stats["residual_mean"])
    residual_std = float(stats["residual_std"])
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x"].to(device)
        residual = batch["residual"].to(device)
        base = batch["base_residual"].to(device)
        target = batch["target"].to(device)
        era = batch["era_ssrd"].to(device)
        mask = batch["mask"].to(device)
        cond = make_cond(x, base)
        refined_residual = refiner_sample(model, cond, base, sample_steps, solver)
        pred = era + refined_residual * residual_std + residual_mean
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
                target_np = target[i, 0].detach().cpu().numpy()
                pred_np = pred[i, 0].detach().cpu().numpy()
                base_np = base_pred[i, 0].detach().cpu().numpy()
                ssim_values.append(core.global_ssim(target_np, pred_np, ssim_data_range))
                base_ssim_values.append(core.global_ssim(target_np, base_np, ssim_data_range))
    return {
        "mae": abs_sum / max(count, 1),
        "rmse": math.sqrt(sq_sum / max(count, 1)),
        "ssim": float(np.mean(ssim_values)) if ssim_values else np.nan,
        "lpips": np.nan,
        "fid": np.nan,
        "base_mae": base_abs_sum / max(count, 1),
        "base_rmse": math.sqrt(base_sq_sum / max(count, 1)),
        "base_ssim": float(np.mean(base_ssim_values)) if base_ssim_values else np.nan,
        "sample_mean_corr": float(pd.Series(target_means).corr(pd.Series(pred_means))) if len(target_means) > 1 else np.nan,
        "base_sample_mean_corr": float(pd.Series(target_means).corr(pd.Series(base_means))) if len(target_means) > 1 else np.nan,
    }


def build_dataset(index_path: Path, metadata: dict, stats: dict, input_channels: str | None, cache_path: Path, max_samples: int | None) -> tuple[CachedBaseDataset, dict, list[str]]:
    manifest, base_metadata = core.build_manifest_from_index(index_path)
    source_x_channels = list(base_metadata["x_channels"])
    source_era_ssrd_idx = source_x_channels.index("era_ssrd")
    x_channels, x_indices, selected_metadata = core.select_input_channels(base_metadata, input_channels)
    selected_metadata["source_x_channels"] = source_x_channels
    selected_metadata["input_channel_indices"] = x_indices
    base_ds = core.TaskBDataset(
        manifest,
        source_era_ssrd_idx,
        x_mean=np.asarray(stats["x_mean"], dtype=np.float32),
        x_std=np.asarray(stats["x_std"], dtype=np.float32),
        residual_mean=stats["residual_mean"],
        residual_std=stats["residual_std"],
        x_indices=x_indices,
    )
    return CachedBaseDataset(base_ds, cache_path, max_samples=max_samples), selected_metadata, x_channels


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = core.select_device(args.device)

    base_ckpt_path = args.base_run_dir / f"{args.base_checkpoint}_model.pt"
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    stats = base_ckpt.get("stats") or json.loads((args.base_run_dir / "normalization_stats.json").read_text())

    train_ds, metadata, x_channels = build_dataset(
        args.train_index, base_ckpt["metadata"], stats, args.input_channels, args.train_base_cache, args.max_train_samples
    )
    val_ds, _val_metadata, _val_channels = build_dataset(
        args.val_index, base_ckpt["metadata"], stats, args.input_channels, args.val_base_cache, args.max_val_samples
    )
    metadata["x_channels"] = x_channels
    metadata["refiner_condition_channels"] = x_channels + ["flowmatch_base_residual_norm"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = make_model(args, len(metadata["refiner_condition_channels"]), device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = {
        "method": "flowmatch_base_self_conditioned_rectified_flow_refiner",
        "args": json_safe(vars(args)),
        "metadata": metadata,
        "base_checkpoint": str(base_ckpt_path),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "strong_features": [
            "frozen FlowMatch best prediction as initial state",
            "U-Net backbone",
            "EMA checkpointing",
            "few-step Heun/Euler refinement",
            "velocity + endpoint + gradient losses",
            "FlowMatch base prediction included as conditioning channel",
        ],
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.out_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    history: list[dict[str, object]] = []
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    resume_path = args.resume_checkpoint
    if resume_path is None and args.resume:
        resume_path = args.out_dir / "last_model.pt"
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        ema_model.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint.get("raw_model", checkpoint["model"]))
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
            patience = pd.to_numeric(history_df.get("epochs_without_improvement", pd.Series(dtype=float)), errors="coerce")
            valid_patience = history_df[patience.notna()]
            if not valid_patience.empty:
                epochs_without_improvement = int(patience.loc[valid_patience.index[-1]])
        print(f"resuming from {resume_path}; next_epoch={start_epoch}", flush=True)

    print(f"device={device}", flush=True)
    print(f"input_channels={x_channels}", flush=True)
    print(f"condition_channels={metadata['refiner_condition_channels']}", flush=True)
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x = batch["x"].to(device)
            residual = batch["residual"].to(device)
            base = batch["base_residual"].to(device)
            mask = batch["mask"].to(device)
            cond = make_cond(x, base)
            bsz = residual.shape[0]
            t = sample_times(bsz, device, args.time_sampling)
            interp = (1.0 - t) * base + t * residual
            if args.train_noise_scale > 0:
                sigma = float(args.train_noise_scale) * torch.sin(math.pi * t).clamp_min(0.0)
                interp = interp + sigma * torch.randn_like(residual)
            target_velocity = residual - base
            pred_velocity = model(interp, cond, t[:, 0, 0, 0] * 999.0)
            velocity_loss = masked_loss(pred_velocity, target_velocity, mask, args.loss)
            endpoint = interp + (1.0 - t) * pred_velocity
            endpoint_loss = masked_loss(endpoint, residual, mask, "l1")
            grad_loss = gradient_l1(endpoint, residual, mask)
            loss = velocity_loss + args.endpoint_loss_weight * endpoint_loss + args.gradient_loss_weight * grad_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            unet.update_ema(ema_model, model, args.ema_decay)
            losses.append(float(loss.item()))

        record: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record["val_loss"] = evaluate_flow_loss(
                ema_model, val_loader, device, args.seed + 1000, args.time_sampling, args.loss, args.train_noise_scale
            )
            metrics = evaluate(ema_model, val_loader, stats, args.sample_steps, args.solver, device, args.ssim_data_range)
            record.update({f"val_{key}": value for key, value in metrics.items()})
            improved = metrics["mae"] < (best_mae - args.early_stopping_min_delta)
            if improved:
                best_mae = metrics["mae"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": ema_model.state_dict(),
                        "raw_model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "metadata": metadata,
                        "stats": stats,
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    args.out_dir / "best_model.pt",
                )
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
        torch.save(
            {
                "model": ema_model.state_dict(),
                "raw_model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "metadata": metadata,
                "stats": stats,
                "epoch": epoch,
            },
            args.out_dir / "last_model.pt",
        )
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

