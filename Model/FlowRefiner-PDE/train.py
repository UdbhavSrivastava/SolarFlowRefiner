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
    parser = argparse.ArgumentParser(description="Train FlowMatch-initialized PDE-Refiner-style denoising refiner.")
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--train-base-cache", type=Path, required=True)
    parser.add_argument("--val-base-cache", type=Path, required=True)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-channels", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-multipliers", type=parse_multipliers, default=parse_multipliers("1,2,4,8"))
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument("--sigma-max", type=float, default=0.35)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--refine-strength", type=float, default=1.0)
    parser.add_argument(
        "--train-state-mode",
        choices=["base_to_target", "target_noise"],
        default="base_to_target",
        help="base_to_target matches inference by corrupting states along FlowMatch-base -> target path.",
    )
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--mse-loss-weight", type=float, default=0.1)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.05)
    parser.add_argument("--base-consistency-weight", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--eval-every", type=int, default=3)
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
    return (loss_x + loss_y) / (mask_dx.sum() + mask_dy.sum()).clamp_min(1.0)


def sigma_schedule(steps: int, sigma_max: float, sigma_min: float, device: torch.device) -> torch.Tensor:
    steps = max(int(steps), 1)
    if steps == 1:
        return torch.tensor([float(sigma_min)], device=device)
    return torch.exp(torch.linspace(math.log(float(sigma_max)), math.log(float(sigma_min)), steps, device=device))


def time_labels(k: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 1:
        return torch.zeros_like(k, dtype=torch.float32)
    return k.float() * (999.0 / float(steps - 1))


def make_cond(x: torch.Tensor, base_residual: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, base_residual], dim=1)


def path_alpha(k: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 1:
        return torch.ones((k.shape[0], 1, 1, 1), device=k.device, dtype=torch.float32)
    return (k.float() / float(steps - 1)).view(-1, 1, 1, 1).clamp(0.0, 1.0)


def make_training_state(
    residual: torch.Tensor,
    base: torch.Tensor,
    sigma: torch.Tensor,
    k: torch.Tensor,
    steps: int,
    mode: str,
    noise: torch.Tensor,
) -> torch.Tensor:
    if mode == "target_noise":
        center = residual
    else:
        alpha = path_alpha(k, steps)
        center = (1.0 - alpha) * base + alpha * residual
    return center + sigma * noise


def make_model(args: argparse.Namespace, cond_channels: int, device: torch.device) -> nn.Module:
    return unet.UNet(
        cond_channels=cond_channels,
        base_channels=int(args.base_channels),
        channel_multipliers=tuple(int(v) for v in args.channel_multipliers),
        res_blocks=int(args.res_blocks),
    ).to(device)


@torch.no_grad()
def pde_refine_sample(
    model: nn.Module,
    cond: torch.Tensor,
    base_residual: torch.Tensor,
    steps: int,
    refine_strength: float,
) -> torch.Tensor:
    model.eval()
    z = base_residual.clone()
    steps = max(int(steps), 1)
    strength = float(refine_strength)
    bsz = z.shape[0]
    for k_idx in range(steps):
        k = torch.full((bsz,), k_idx, device=z.device, dtype=torch.long)
        pred_clean = model(z, cond, time_labels(k, steps))
        z = z + strength * (pred_clean - z)
    return z


@torch.no_grad()
def evaluate_denoising_loss(
    model: nn.Module,
    loader: DataLoader,
    sigmas: torch.Tensor,
    device: torch.device,
    seed: int,
    loss_name: str,
    train_state_mode: str,
) -> float:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    loss_sum = 0.0
    count = 0
    steps = int(sigmas.numel())
    for batch in tqdm(loader, desc="val_loss", leave=False):
        x = batch["x"].to(device)
        residual = batch["residual"].to(device)
        base = batch["base_residual"].to(device)
        mask = batch["mask"].to(device)
        cond = make_cond(x, base)
        bsz = residual.shape[0]
        k = torch.randint(0, steps, (bsz,), device=device, generator=generator)
        sigma = sigmas[k].view(bsz, 1, 1, 1)
        noise = torch.randn(residual.shape, device=device, generator=generator)
        noisy = make_training_state(residual, base, sigma, k, steps, train_state_mode, noise)
        pred_clean = model(noisy, cond, time_labels(k, steps))
        values = torch.square(pred_clean - residual) if loss_name == "mse" else torch.abs(pred_clean - residual)
        loss_sum += float(values.mul(mask).sum().item())
        count += int(mask.sum().item())
    return loss_sum / max(count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    stats: dict,
    steps: int,
    refine_strength: float,
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
        base = batch["base_residual"].to(device)
        target = batch["target"].to(device)
        era = batch["era_ssrd"].to(device)
        mask = batch["mask"].to(device)
        cond = make_cond(x, base)
        refined_residual = pde_refine_sample(model, cond, base, steps, refine_strength)
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


def build_dataset(index_path: Path, stats: dict, input_channels: str | None, cache_path: Path, max_samples: int | None) -> tuple[CachedBaseDataset, dict, list[str]]:
    manifest, metadata = core.build_manifest_from_index(index_path)
    source_x_channels = list(metadata["x_channels"])
    source_era_ssrd_idx = source_x_channels.index("era_ssrd")
    x_channels, x_indices, selected_metadata = core.select_input_channels(metadata, input_channels)
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
        args.train_index, stats, args.input_channels, args.train_base_cache, args.max_train_samples
    )
    val_ds, _val_metadata, _val_channels = build_dataset(
        args.val_index, stats, args.input_channels, args.val_base_cache, args.max_val_samples
    )
    metadata["x_channels"] = x_channels
    metadata["refiner_condition_channels"] = x_channels + ["flowmatch_base_residual_norm"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = make_model(args, len(metadata["refiner_condition_channels"]), device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sigmas = sigma_schedule(args.refinement_steps, args.sigma_max, args.sigma_min, device)

    run_config = {
        "method": "flowmatch_initialized_pde_refiner_base_to_target_denoising",
        "args": json_safe(vars(args)),
        "metadata": metadata,
        "base_checkpoint": str(base_ckpt_path),
        "sigma_schedule": [float(v) for v in sigmas.detach().cpu().tolist()],
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "paper_like_features": [
            "iterative denoising refinement",
            "exponentially decreasing sigma schedule",
            "U-Net backbone",
            "frozen FlowMatch initial prediction",
            "base-to-target path denoising objective",
            "EMA checkpointing",
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
    print(f"sigmas={[float(v) for v in sigmas.detach().cpu().tolist()]}", flush=True)
    print(f"train_state_mode={args.train_state_mode}", flush=True)
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
            k = torch.randint(0, args.refinement_steps, (bsz,), device=device)
            sigma = sigmas[k].view(bsz, 1, 1, 1)
            noisy = make_training_state(
                residual,
                base,
                sigma,
                k,
                args.refinement_steps,
                args.train_state_mode,
                torch.randn_like(residual),
            )
            pred_clean = model(noisy, cond, time_labels(k, args.refinement_steps))
            primary = masked_loss(pred_clean, residual, mask, args.loss)
            mse = masked_loss(pred_clean, residual, mask, "mse")
            grad = gradient_l1(pred_clean, residual, mask)
            loss = primary + args.mse_loss_weight * mse + args.gradient_loss_weight * grad
            if args.base_consistency_weight > 0:
                base_improved = torch.minimum(torch.abs(pred_clean - residual), torch.abs(base - residual))
                consistency = torch.relu(torch.abs(pred_clean - residual) - base_improved).mul(mask).sum() / mask.sum().clamp_min(1.0)
                loss = loss + args.base_consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            unet.update_ema(ema_model, model, args.ema_decay)
            losses.append(float(loss.item()))

        record: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record["val_loss"] = evaluate_denoising_loss(
                ema_model, val_loader, sigmas, device, args.seed + 1000, args.loss, args.train_state_mode
            )
            metrics = evaluate(
                ema_model,
                val_loader,
                stats,
                args.refinement_steps,
                args.refine_strength,
                device,
                args.ssim_data_range,
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

