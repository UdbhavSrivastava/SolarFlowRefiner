from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


try:
    from . import core
except ImportError:
    import core


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def parse_multipliers(value: str) -> tuple[int, ...]:
    vals = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if len(vals) < 2:
        raise argparse.ArgumentTypeError("Use at least two channel multipliers, e.g. 1,2,4,8")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-style conditional DDPM residual model without Prithvi.")
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
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-6)
    parser.add_argument("--beta-end", type=float, default=1e-2)
    parser.add_argument("--sample-steps", type=int, default=50)
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


def valid_group_count(channels: int, max_groups: int = 16) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(valid_group_count(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(valid_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(self.act(temb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        self.heads = max(1, min(heads, channels))
        while channels % self.heads != 0:
            self.heads -= 1
        self.norm = nn.GroupNorm(valid_group_count(channels), channels)
        self.attn = nn.MultiheadAttention(channels, self.heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(y)


class UNet(nn.Module):
    def __init__(
        self,
        cond_channels: int,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        res_blocks: int = 2,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        channels = [base_channels * mult for mult in channel_multipliers]
        self.in_conv = nn.Conv2d(cond_channels + 1, channels[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, time_dim))
                in_ch = out_ch
            self.enc_blocks.append(blocks)
            if level < len(channels) - 1:
                self.downs.append(nn.Conv2d(in_ch, channels[level + 1], 4, stride=2, padding=1))
                in_ch = channels[level + 1]

        self.mid1 = ResBlock(channels[-1], channels[-1], time_dim)
        self.mid_attn = SelfAttention2d(channels[-1], heads=8)
        self.mid2 = ResBlock(channels[-1], channels[-1], time_dim)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for level in reversed(range(len(channels) - 1)):
            self.ups.append(nn.ConvTranspose2d(channels[level + 1], channels[level], 4, stride=2, padding=1))
            blocks = nn.ModuleList()
            in_ch = channels[level] * 2
            for block_idx in range(res_blocks):
                out_ch = channels[level]
                blocks.append(ResBlock(in_ch, out_ch, time_dim))
                in_ch = out_ch
            self.dec_blocks.append(blocks)

        self.out = nn.Sequential(
            nn.GroupNorm(valid_group_count(channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], 1, 3, padding=1),
        )

    def forward(self, noisy_residual: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.in_conv(torch.cat([noisy_residual, cond], dim=1))
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            if level < len(self.downs):
                h = self.downs[level](h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        # The deepest encoder output is already represented by h; decoder skips
        # start at the next finer resolution.
        skips.pop()
        for up, blocks in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, temb)
        return self.out(h)


def make_schedule(timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mul(mask).sum() / mask.sum().clamp_min(1.0)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


@torch.no_grad()
def evaluate_noise_loss(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    timesteps: int,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    abs_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="val_loss", leave=False):
        cond = batch["x"].to(device)
        residual = batch["residual"].to(device)
        mask = batch["mask"].to(device)
        bsz = residual.shape[0]
        t = torch.randint(0, timesteps, (bsz,), device=device, generator=generator)
        noise = torch.randn(residual.shape, device=device, generator=generator)
        ab = schedule.alpha_bars[t].view(bsz, 1, 1, 1)
        noisy = torch.sqrt(ab) * residual + torch.sqrt(1.0 - ab) * noise
        pred_noise = model(noisy, cond, t)
        abs_sum += float(((pred_noise - noise).abs() * mask).sum().item())
        count += int(mask.sum().item())
    return abs_sum / max(count, 1)


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    cond: torch.Tensor,
    schedule: DiffusionSchedule,
    sample_steps: int,
    residual_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    model.eval()
    device = cond.device
    steps = torch.linspace(len(schedule.alpha_bars) - 1, 0, sample_steps, device=device).long()
    z = torch.randn(residual_shape, device=device)
    for step_idx, t_value in enumerate(steps):
        t = torch.full((residual_shape[0],), int(t_value.item()), device=device, dtype=torch.long)
        eps = model(z, cond, t)
        ab = schedule.alpha_bars[t_value].view(1, 1, 1, 1)
        x0 = (z - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        if step_idx == len(steps) - 1:
            z = x0
        else:
            next_t = steps[step_idx + 1]
            ab_next = schedule.alpha_bars[next_t].view(1, 1, 1, 1)
            z = torch.sqrt(ab_next) * x0 + torch.sqrt(1 - ab_next) * eps
    return z


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    stats: dict,
    sample_steps: int,
    device: torch.device,
    ssim_data_range: float,
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
    for batch in tqdm(loader, desc="eval", leave=False):
        cond = batch["x"].to(device)
        target = batch["target"].to(device)
        era = batch["era_ssrd"].to(device)
        mask = batch["mask"].to(device)
        residual_norm = ddim_sample(model, cond, schedule, sample_steps, batch["residual"].shape)
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
        "method": "paper_style_ddpm_no_prithvi",
        "args": json_safe(vars(args)),
        "metadata": metadata,
        "train_samples": int(len(train_manifest)),
        "val_samples": int(len(val_manifest)),
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

    model = UNet(
        cond_channels=len(x_channels),
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        res_blocks=args.res_blocks,
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = make_schedule(args.timesteps, args.beta_start, args.beta_end, device)

    history: list[dict[str, object]] = []
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    resume_path = args.resume_checkpoint
    if resume_path is None and args.resume:
        resume_path = args.out_dir / "last_model.pt"
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("raw_model", checkpoint["model"]))
        ema_model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        history_path = args.out_dir / "training_history.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            history_df = history_df[history_df["epoch"] < start_epoch]
            history = history_df.to_dict("records")
            if not history_df.empty:
                last_record = history_df.iloc[-1]
                best_mae = float(last_record.get("best_val_mae", history_df["val_mae"].min()))
                best_epoch = int(last_record.get("best_epoch", history_df.loc[history_df["val_mae"].idxmin(), "epoch"]))
                epochs_without_improvement = int(last_record.get("epochs_without_improvement", 0))
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
            t = torch.randint(0, args.timesteps, (bsz,), device=device)
            noise = torch.randn_like(residual)
            ab = schedule.alpha_bars[t].view(bsz, 1, 1, 1)
            noisy = torch.sqrt(ab) * residual + torch.sqrt(1.0 - ab) * noise
            pred_noise = model(noisy, cond, t)
            loss = masked_l1(pred_noise, noise, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model, args.ema_decay)
            losses.append(float(loss.item()))

        record: dict[str, object] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record["val_loss"] = evaluate_noise_loss(ema_model, val_loader, schedule, args.timesteps, device, args.seed + 1000)
            metrics = evaluate(ema_model, val_loader, schedule, stats, args.sample_steps, device, args.ssim_data_range)
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
