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
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils import core
from utils import unet


def parse_multipliers(value: str) -> tuple[int, ...]:
    vals = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if len(vals) < 2:
        raise argparse.ArgumentTypeError("Use at least two channel multipliers, e.g. 1,2,4,8")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced conditional FlowMatch residual model.")
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--input-channels", default=None, help="Default/all uses ERA + SolarCube auxiliary channels.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-multipliers", type=parse_multipliers, default=parse_multipliers("1,2,4,8"))
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--time-sampling", choices=["logit_normal", "uniform"], default="logit_normal")
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
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


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        values = torch.square(pred - target)
    else:
        values = torch.abs(pred - target)
    return values.mul(mask).sum() / mask.sum().clamp_min(1.0)


def sample_times(batch_size: int, device: torch.device, mode: str, generator: torch.Generator | None = None) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand((batch_size, 1, 1, 1), device=device, generator=generator)
    else:
        t = torch.sigmoid(torch.randn((batch_size, 1, 1, 1), device=device, generator=generator))
    return t.clamp(1e-4, 1.0 - 1e-4)


@torch.no_grad()
def flow_sample(
    model: nn.Module,
    cond: torch.Tensor,
    residual_shape: tuple[int, int, int, int],
    steps: int,
    solver: str,
    seed: int | None = None,
) -> torch.Tensor:
    model.eval()
    device = cond.device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))
    z = torch.randn(residual_shape, device=device, generator=generator)
    n_steps = max(int(steps), 1)
    dt = 1.0 / float(n_steps)
    for step in range(n_steps):
        t0_float = step / float(n_steps)
        t1_float = (step + 1) / float(n_steps)
        t0 = torch.full((residual_shape[0],), t0_float * 999.0, device=device)
        v0 = model(z, cond, t0)
        if solver == "heun" and step < n_steps - 1:
            z_euler = z + dt * v0
            t1 = torch.full((residual_shape[0],), t1_float * 999.0, device=device)
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
) -> float:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    loss_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="val_loss", leave=False):
        cond = batch["x"].to(device)
        residual = batch["residual"].to(device)
        mask = batch["mask"].to(device)
        bsz = residual.shape[0]
        t = sample_times(bsz, device, time_sampling, generator)
        noise = torch.randn(residual.shape, device=device, generator=generator)
        x_t = (1.0 - t) * noise + t * residual
        target_velocity = residual - noise
        pred_velocity = model(x_t, cond, t[:, 0, 0, 0] * 999.0)
        if loss_name == "mse":
            values = torch.square(pred_velocity - target_velocity)
        else:
            values = torch.abs(pred_velocity - target_velocity)
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
    seed: int,
) -> dict[str, float]:
    abs_sum = sq_sum = base_abs_sum = base_sq_sum = 0.0
    count = 0
    target_means: list[float] = []
    pred_means: list[float] = []
    base_means: list[float] = []
    ssim_values: list[float] = []
    baseline_ssim_values: list[float] = []
    residual_mean = float(stats["residual_mean"])
    residual_std = float(stats["residual_std"])
    for batch_idx, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        cond = batch["x"].to(device)
        target = batch["target"].to(device)
        era = batch["era_ssrd"].to(device)
        mask = batch["mask"].to(device)
        residual_norm = flow_sample(model, cond, batch["residual"].shape, sample_steps, solver, seed + batch_idx)
        pred = era + residual_norm * residual_std + residual_mean
        diff = (pred - target) * mask
        base_diff = (era - target) * mask
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
                base_means.append(float(era[i][sample_mask].mean().item()))
                target_np = target[i, 0].detach().cpu().numpy()
                pred_np = pred[i, 0].detach().cpu().numpy()
                era_np = era[i, 0].detach().cpu().numpy()
                ssim_values.append(core.global_ssim(target_np, pred_np, ssim_data_range))
                baseline_ssim_values.append(core.global_ssim(target_np, era_np, ssim_data_range))
    return {
        "mae": abs_sum / max(count, 1),
        "rmse": math.sqrt(sq_sum / max(count, 1)),
        "ssim": float(np.mean(ssim_values)) if ssim_values else np.nan,
        "lpips": np.nan,
        "fid": np.nan,
        "baseline_mae": base_abs_sum / max(count, 1),
        "baseline_rmse": math.sqrt(base_sq_sum / max(count, 1)),
        "baseline_ssim": float(np.mean(baseline_ssim_values)) if baseline_ssim_values else np.nan,
        "baseline_lpips": np.nan,
        "baseline_fid": np.nan,
        "sample_mean_corr": float(pd.Series(target_means).corr(pd.Series(pred_means))) if len(target_means) > 1 else np.nan,
        "baseline_sample_mean_corr": float(pd.Series(target_means).corr(pd.Series(base_means))) if len(target_means) > 1 else np.nan,
    }


def make_model(args: argparse.Namespace, cond_channels: int, device: torch.device) -> nn.Module:
    return unet.UNet(
        cond_channels=cond_channels,
        base_channels=int(args.base_channels),
        channel_multipliers=tuple(int(v) for v in args.channel_multipliers),
        res_blocks=int(args.res_blocks),
    ).to(device)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = core.select_device(args.device)
    print(f"device={device}", flush=True)
    train_manifest, metadata = core.build_manifest_from_index(args.train_index)
    val_manifest, _ = core.build_manifest_from_index(args.val_index)
    source_x_channels = list(metadata["x_channels"])
    source_era_ssrd_idx = source_x_channels.index("era_ssrd")
    x_channels, x_indices, metadata = core.select_input_channels(metadata, args.input_channels)
    print(f"input_channels={x_channels}", flush=True)

    run_config = {
        "method": "advanced_conditional_flow_matching_rectified_flow",
        "args": json_safe(vars(args)),
        "metadata": metadata,
        "train_samples": int(len(train_manifest)),
        "val_samples": int(len(val_manifest)),
        "advanced_features": [
            "all conditioning channels by default",
            "U-Net backbone",
            "EMA checkpointing",
            "logit-normal time sampling",
            "Heun ODE sampler",
            "fixed-seed deterministic validation sampling",
        ],
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    stats_path = args.stats_path if args.stats_path is not None else args.out_dir / "normalization_stats.json"
    stats = core.load_or_compute_stats(
        train_manifest,
        source_era_ssrd_idx,
        stats_path,
        mirror_path=args.out_dir / "normalization_stats.json",
        x_indices=x_indices,
    )
    train_ds = core.TaskBDataset(
        train_manifest,
        source_era_ssrd_idx,
        x_mean=np.asarray(stats["x_mean"], dtype=np.float32),
        x_std=np.asarray(stats["x_std"], dtype=np.float32),
        residual_mean=stats["residual_mean"],
        residual_std=stats["residual_std"],
        max_samples=args.max_train_samples,
        x_indices=x_indices,
    )
    val_ds = core.TaskBDataset(
        val_manifest,
        source_era_ssrd_idx,
        x_mean=np.asarray(stats["x_mean"], dtype=np.float32),
        x_std=np.asarray(stats["x_std"], dtype=np.float32),
        residual_mean=stats["residual_mean"],
        residual_std=stats["residual_std"],
        max_samples=args.max_val_samples,
        x_indices=x_indices,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args, len(x_channels), device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            if not history_df.empty:
                if "val_mae" in history_df:
                    val_mae = pd.to_numeric(history_df["val_mae"], errors="coerce")
                    valid_val_rows = history_df[val_mae.notna()]
                    if not valid_val_rows.empty:
                        best_idx = val_mae.loc[valid_val_rows.index].idxmin()
                        best_mae = float(history_df.loc[best_idx, "val_mae"])
                        best_epoch = int(history_df.loc[best_idx, "epoch"])
                if "epochs_without_improvement" in history_df:
                    patience_values = pd.to_numeric(history_df["epochs_without_improvement"], errors="coerce")
                    valid_patience_rows = history_df[patience_values.notna()]
                    if not valid_patience_rows.empty:
                        epochs_without_improvement = int(patience_values.loc[valid_patience_rows.index[-1]])
        print(f"resuming from {resume_path}; next_epoch={start_epoch}", flush=True)

    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            cond = batch["x"].to(device)
            residual = batch["residual"].to(device)
            mask = batch["mask"].to(device)
            bsz = residual.shape[0]
            t = sample_times(bsz, device, args.time_sampling)
            noise = torch.randn_like(residual)
            x_t = (1.0 - t) * noise + t * residual
            target_velocity = residual - noise
            pred_velocity = model(x_t, cond, t[:, 0, 0, 0] * 999.0)
            loss = masked_loss(pred_velocity, target_velocity, mask, args.loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            unet.update_ema(ema_model, model, args.ema_decay)
            losses.append(float(loss.item()))

        record: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record["val_loss"] = evaluate_flow_loss(
                ema_model, val_loader, device, args.seed + 1000, args.time_sampling, args.loss
            )
            metrics = evaluate(
                ema_model,
                val_loader,
                stats,
                args.sample_steps,
                args.solver,
                device,
                args.ssim_data_range,
                args.seed + 2000,
            )
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
