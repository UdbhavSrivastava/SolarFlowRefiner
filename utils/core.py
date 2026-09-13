from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .metrics import global_ssim
except ImportError:
    from metrics import global_ssim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "diffusion_unet_residual"
NPZ_READ_RETRIES = 8
NPZ_READ_EXCEPTIONS = (OSError, EOFError, zipfile.BadZipFile)


def repo_root() -> Path:
    return PROJECT_ROOT


def portable_path(value: str | Path) -> Path:
    text = str(value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute():
        return path
    return repo_root() / path


def split_chunk_path(month_dir_value: str | Path, chunk_value: str | Path) -> Path:
    chunk_text = str(chunk_value).replace("\\", "/")
    return portable_path(month_dir_value) / Path(chunk_text)


def retry_npz_read_message(path: Path, attempt: int, exc: BaseException) -> None:
    delay = min(2.0 * attempt, 10.0)
    tqdm.write(
        f"Retrying npz read after {type(exc).__name__}: {path} "
        f"(attempt {attempt + 1}/{NPZ_READ_RETRIES}, sleep {delay:.1f}s)"
    )
    time.sleep(delay)


def read_stats_rows_with_retry(path: Path, local_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for attempt in range(1, NPZ_READ_RETRIES + 1):
        try:
            with np.load(path) as chunk:
                x = chunk["x"][local_rows].astype(np.float64)
                y = chunk["y"][local_rows, 0].astype(np.float64)
                mask = chunk["valid_mask"][local_rows, 0].astype(bool)
            return x, y, mask
        except NPZ_READ_EXCEPTIONS as exc:
            if attempt == NPZ_READ_RETRIES:
                raise
            retry_npz_read_message(path, attempt, exc)
    raise RuntimeError(f"Failed to read {path}")


def load_or_compute_stats(
    train_manifest: pd.DataFrame,
    era_ssrd_idx: int,
    stats_path: Path,
    mirror_path: Path | None = None,
    x_indices: list[int] | None = None,
) -> dict:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path.exists():
        print(f"Using normalization stats: {stats_path}", flush=True)
        stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
    else:
        print(f"Computing normalization stats: {stats_path}", flush=True)
        stats = compute_stats(train_manifest, era_ssrd_idx, x_indices=x_indices)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if mirror_path is not None and mirror_path.resolve() != stats_path.resolve():
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        if not mirror_path.exists():
            mirror_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def parse_months(value: str) -> list[int]:
    months: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            months.extend(range(int(start), int(end) + 1))
        else:
            months.append(int(part))
    return months


def parse_channel_names(value: str | None) -> list[str] | None:
    if value is None or not value.strip() or value.strip().lower() in {"all", "*"}:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def select_input_channels(metadata: dict, requested: str | None) -> tuple[list[str], list[int], dict]:
    source_channels = list(metadata["x_channels"])
    requested_channels = parse_channel_names(requested)
    if requested_channels is None:
        selected_channels = source_channels
    else:
        missing = [name for name in requested_channels if name not in source_channels]
        if missing:
            raise ValueError(f"Requested input channels not found in metadata: {missing}")
        selected_channels = requested_channels
    indices = [source_channels.index(name) for name in selected_channels]
    selected_metadata = dict(metadata)
    selected_metadata["source_x_channels"] = source_channels
    selected_metadata["x_channels"] = selected_channels
    selected_metadata["input_channel_indices"] = indices
    return selected_channels, indices, selected_metadata


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional diffusion U-Net for SolarCube SSR residuals.")
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tile-id", type=int, default=1)
    parser.add_argument("--train-index", type=Path, default=None)
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--train-months", default="1-9")
    parser.add_argument("--val-months", default="10-12")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument(
        "--input-channels",
        default=None,
        help="Comma-separated x channel names to use. Default uses all channels. Example: era_ssrd,era_ssr,era_ssrdc,era_fdir,era_cdir",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last_model.pt in --out-dir and continue at the next epoch.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path to resume from. Overrides --resume if provided.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many evaluated epochs without val MAE improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum val MAE improvement required to reset early-stopping patience.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--ssim-data-range", type=float, default=1000.0)
    return parser.parse_args()


def month_dir(root: Path, tile_id: int, month: int) -> Path:
    return root / f"tile_{tile_id:02d}" / f"month_{month:02d}"


def build_manifest(root: Path, tile_id: int, months: list[int]) -> tuple[pd.DataFrame, dict]:
    frames = []
    metadata: dict | None = None
    for month in months:
        directory = month_dir(root, tile_id, month)
        if metadata is None:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
        index = pd.read_csv(directory / "samples_index.csv")
        index["month"] = month
        index["month_dir"] = str(directory)
        frames.append(index)
    if metadata is None:
        raise RuntimeError("No metadata found")
    return pd.concat(frames, ignore_index=True), metadata


def build_manifest_from_index(index_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(index_path)
    if "month_dir" not in manifest.columns:
        raise ValueError(f"Split index must contain a month_dir column: {index_path}")
    if manifest.empty:
        raise ValueError(f"Split index is empty: {index_path}")

    metadata_path = portable_path(manifest.iloc[0]["month_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    return manifest, metadata


class TaskBDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        era_ssrd_idx: int,
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
        residual_mean: float = 0.0,
        residual_std: float = 1.0,
        max_samples: int | None = None,
        x_indices: list[int] | None = None,
    ) -> None:
        if max_samples is not None and len(manifest) > max_samples:
            manifest = manifest.sample(max_samples, random_state=13).sort_index()
        self.manifest = manifest.reset_index(drop=True)
        self.era_ssrd_idx = era_ssrd_idx
        self.x_mean = x_mean
        self.x_std = x_std
        self.residual_mean = float(residual_mean)
        self.residual_std = float(max(residual_std, 1e-6))
        self.x_indices = x_indices
        self._cache_path: Path | None = None
        self._cache = None

    def __len__(self) -> int:
        return len(self.manifest)

    def _drop_cache(self) -> None:
        if self._cache is not None:
            self._cache.close()
        self._cache = None
        self._cache_path = None

    def _load_chunk(self, path: Path):
        if self._cache_path != path:
            self._drop_cache()
            for attempt in range(1, NPZ_READ_RETRIES + 1):
                try:
                    self._cache = np.load(path)
                    self._cache_path = path
                    break
                except NPZ_READ_EXCEPTIONS as exc:
                    if attempt == NPZ_READ_RETRIES:
                        raise
                    retry_npz_read_message(path, attempt, exc)
        return self._cache

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        chunk_path = split_chunk_path(row["month_dir"], row["chunk"])
        local_row = int(row["row"])
        for attempt in range(1, NPZ_READ_RETRIES + 1):
            try:
                chunk = self._load_chunk(chunk_path)
                x = chunk["x"][local_row].astype(np.float32)
                y = chunk["y"][local_row, 0].astype(np.float32)
                mask = chunk["valid_mask"][local_row, 0].astype(np.float32)
                break
            except NPZ_READ_EXCEPTIONS as exc:
                self._drop_cache()
                if attempt == NPZ_READ_RETRIES:
                    raise
                retry_npz_read_message(chunk_path, attempt, exc)
        era_ssrd = x[self.era_ssrd_idx].astype(np.float32)
        residual = y - era_ssrd
        if self.x_indices is not None:
            x = x[self.x_indices]

        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean[:, None, None]) / self.x_std[:, None, None]
        residual = (residual - self.residual_mean) / self.residual_std

        return {
            "x": torch.from_numpy(x),
            "residual": torch.from_numpy(residual[None, :, :]),
            "target": torch.from_numpy(y[None, :, :]),
            "era_ssrd": torch.from_numpy(era_ssrd[None, :, :]),
            "mask": torch.from_numpy(mask[None, :, :]),
        }

    def close(self) -> None:
        self._drop_cache()


def compute_stats(manifest: pd.DataFrame, era_ssrd_idx: int, x_indices: list[int] | None = None) -> dict:
    x_sum = None
    x_sumsq = None
    x_count = 0
    residual_sum = 0.0
    residual_sumsq = 0.0
    residual_count = 0

    for chunk_key, rows in tqdm(manifest.groupby(["month_dir", "chunk"], sort=False), desc="stats"):
        month_dir_value, chunk_rel = chunk_key
        local_rows = rows["row"].to_numpy(dtype=np.int64)
        x, y, mask = read_stats_rows_with_retry(split_chunk_path(month_dir_value, chunk_rel), local_rows)
        residual = y - x[:, era_ssrd_idx]
        if x_indices is not None:
            x = x[:, x_indices]
        if x_sum is None:
            x_sum = np.zeros(x.shape[1], dtype=np.float64)
            x_sumsq = np.zeros(x.shape[1], dtype=np.float64)
        x_sum += x.sum(axis=(0, 2, 3))
        x_sumsq += np.square(x).sum(axis=(0, 2, 3))
        x_count += int(x.shape[0] * x.shape[2] * x.shape[3])

        values = residual[mask]
        residual_sum += float(values.sum(dtype=np.float64))
        residual_sumsq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        residual_count += int(values.size)

    assert x_sum is not None and x_sumsq is not None
    x_mean = x_sum / x_count
    x_var = np.maximum(x_sumsq / x_count - x_mean * x_mean, 1e-6)
    residual_mean = residual_sum / residual_count
    residual_var = max(residual_sumsq / residual_count - residual_mean * residual_mean, 1e-6)
    return {
        "x_mean": x_mean.astype(float).tolist(),
        "x_std": np.sqrt(x_var).astype(float).tolist(),
        "residual_mean": float(residual_mean),
        "residual_std": float(math.sqrt(residual_var)),
    }


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        groups1 = valid_group_count(in_ch)
        groups2 = valid_group_count(out_ch)
        self.norm1 = nn.GroupNorm(groups1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(self.act(temb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


def valid_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DiffusionUNet(nn.Module):
    def __init__(self, cond_channels: int, base: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        in_ch = cond_channels + 1
        self.in_block = ResBlock(in_ch, base, time_dim)
        self.down1 = nn.Sequential(nn.Conv2d(base, base * 2, 4, stride=2, padding=1))
        self.rb1 = ResBlock(base * 2, base * 2, time_dim)
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1))
        self.rb2 = ResBlock(base * 4, base * 4, time_dim)
        self.down3 = nn.Sequential(nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1))
        self.rb3 = ResBlock(base * 8, base * 8, time_dim)
        self.mid = nn.Sequential()
        self.mid1 = ResBlock(base * 8, base * 8, time_dim)
        self.mid2 = ResBlock(base * 8, base * 8, time_dim)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1)
        self.urb3 = ResBlock(base * 8, base * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.urb2 = ResBlock(base * 4, base * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.urb1 = ResBlock(base * 2, base, time_dim)
        self.out = nn.Sequential(nn.GroupNorm(min(8, base), base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, noisy_residual: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        x = torch.cat([noisy_residual, cond], dim=1)
        s0 = self.in_block(x, temb)
        s1 = self.rb1(self.down1(s0), temb)
        s2 = self.rb2(self.down2(s1), temb)
        s3 = self.rb3(self.down3(s2), temb)
        h = self.mid2(self.mid1(s3, temb), temb)
        h = self.urb3(torch.cat([self.up3(h), s2], dim=1), temb)
        h = self.urb2(torch.cat([self.up2(h), s1], dim=1), temb)
        h = self.urb1(torch.cat([self.up1(h), s0], dim=1), temb)
        return self.out(h)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_schedule(timesteps: int, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        test = torch.empty(1, device="cuda")
        _ = test + 1
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as exc:
        print(f"CUDA is available but unusable in this PyTorch build; falling back to CPU: {exc}", flush=True)
        return torch.device("cpu")


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.square(pred - target).mul(mask).sum() / mask.sum().clamp_min(1.0)


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
    sq_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="val_loss", leave=False):
        cond = batch["x"].to(device)
        residual = batch["residual"].to(device)
        mask = batch["mask"].to(device)
        bsz = residual.shape[0]
        t = torch.randint(0, timesteps, (bsz,), device=device, generator=generator)
        noise = torch.randn(residual.shape, device=device, generator=generator)
        ab = schedule.alpha_bars[t].view(bsz, 1, 1, 1)
        noisy = torch.sqrt(ab) * residual + torch.sqrt(1 - ab) * noise
        pred_noise = model(noisy, cond, t)
        diff = (pred_noise - noise) * mask
        sq_sum += float(torch.square(diff).sum().item())
        count += int(mask.sum().item())
    return sq_sum / max(count, 1)


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    stats: dict,
    sample_steps: int,
    device: torch.device,
    ssim_data_range: float,
) -> dict[str, float]:
    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    base_abs_sum = 0.0
    base_sq_sum = 0.0
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
                ssim_values.append(global_ssim(target_np, pred_np, ssim_data_range))
                baseline_ssim_values.append(global_ssim(target_np, era_np, ssim_data_range))
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
        "baseline_sample_mean_corr": float(pd.Series(target_means).corr(pd.Series(base_means)))
        if len(target_means) > 1
        else np.nan,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"device={device}", flush=True)

    if args.train_index is not None or args.val_index is not None:
        if args.train_index is None or args.val_index is None:
            raise ValueError("--train-index and --val-index must be provided together")
        train_manifest, metadata = build_manifest_from_index(args.train_index)
        val_manifest, _metadata = build_manifest_from_index(args.val_index)
    else:
        train_months = parse_months(args.train_months)
        val_months = parse_months(args.val_months)
        train_manifest, metadata = build_manifest(args.preprocess_root, args.tile_id, train_months)
        val_manifest, _metadata = build_manifest(args.preprocess_root, args.tile_id, val_months)
    source_x_channels = list(metadata["x_channels"])
    source_era_ssrd_idx = source_x_channels.index("era_ssrd")
    x_channels, x_indices, metadata = select_input_channels(metadata, args.input_channels)
    era_ssrd_idx = x_channels.index("era_ssrd")
    print(f"input_channels={x_channels}", flush=True)
    run_config = {
        "args": json_safe(vars(args)),
        "metadata": metadata,
        "train_samples": int(len(train_manifest)),
        "val_samples": int(len(val_manifest)),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    stats_path = args.stats_path if args.stats_path is not None else args.out_dir / "normalization_stats.json"
    stats = load_or_compute_stats(
        train_manifest,
        source_era_ssrd_idx,
        stats_path,
        mirror_path=args.out_dir / "normalization_stats.json",
        x_indices=x_indices,
    )

    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    train_ds = TaskBDataset(
        train_manifest,
        era_ssrd_idx,
        x_mean=x_mean,
        x_std=x_std,
        residual_mean=stats["residual_mean"],
        residual_std=stats["residual_std"],
        max_samples=args.max_train_samples,
        x_indices=x_indices,
    )
    val_ds = TaskBDataset(
        val_manifest,
        era_ssrd_idx,
        x_mean=x_mean,
        x_std=x_std,
        residual_mean=stats["residual_mean"],
        residual_std=stats["residual_std"],
        max_samples=args.max_val_samples,
        x_indices=x_indices,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DiffusionUNet(cond_channels=len(x_channels), base=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = make_schedule(args.timesteps, device)
    history = []
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    start_epoch = 1

    resume_path = args.resume_checkpoint
    if resume_path is None and args.resume:
        resume_path = args.out_dir / "last_model.pt"
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Resume checkpoint has no optimizer state; continuing with a fresh Adam optimizer.", flush=True)
        loaded_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = loaded_epoch + 1
        history_path = args.out_dir / "training_history.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            history_df = history_df[history_df["epoch"] < start_epoch]
            history = history_df.to_dict("records")
            if not history_df.empty:
                last_record = history_df.iloc[-1]
                if "best_val_mae" in history_df.columns and pd.notna(last_record.get("best_val_mae")):
                    best_mae = float(last_record["best_val_mae"])
                elif "val_mae" in history_df.columns:
                    best_mae = float(history_df["val_mae"].min())
                if "best_epoch" in history_df.columns and pd.notna(last_record.get("best_epoch")):
                    best_epoch = int(last_record["best_epoch"])
                elif "val_mae" in history_df.columns:
                    best_epoch = int(history_df.loc[history_df["val_mae"].idxmin(), "epoch"])
                if "epochs_without_improvement" in history_df.columns and pd.notna(last_record.get("epochs_without_improvement")):
                    epochs_without_improvement = int(last_record["epochs_without_improvement"])
        print(
            f"Resuming from {resume_path}; loaded_epoch={loaded_epoch}; next_epoch={start_epoch}; "
            f"best_epoch={best_epoch}; best_val_mae={best_mae}",
            flush=True,
        )
        if start_epoch > args.epochs:
            print(f"Checkpoint epoch {loaded_epoch} is already >= requested epochs {args.epochs}. Nothing to do.", flush=True)
            train_ds.close()
            val_ds.close()
            return 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            cond = batch["x"].to(device)
            residual = batch["residual"].to(device)
            mask = batch["mask"].to(device)
            bsz = residual.shape[0]
            t = torch.randint(0, args.timesteps, (bsz,), device=device)
            noise = torch.randn_like(residual)
            ab = schedule.alpha_bars[t].view(bsz, 1, 1, 1)
            noisy = torch.sqrt(ab) * residual + torch.sqrt(1 - ab) * noise
            pred_noise = model(noisy, cond, t)
            loss = masked_mse(pred_noise, noise, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record["val_loss"] = evaluate_noise_loss(
                model,
                val_loader,
                schedule,
                args.timesteps,
                device,
                seed=args.seed + 1000,
            )
            metrics = evaluate(model, val_loader, schedule, stats, args.sample_steps, device, args.ssim_data_range)
            record.update({f"val_{k}": v for k, v in metrics.items()})
            improved = metrics["mae"] < (best_mae - args.early_stopping_min_delta)
            if improved:
                best_mae = metrics["mae"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "metadata": metadata,
                        "stats": stats,
                        "optimizer": optimizer.state_dict(),
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
            log_file.write(json.dumps(record) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "metadata": metadata,
                "stats": stats,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            args.out_dir / "last_model.pt",
        )
        print(record, flush=True)
        if stopped_early:
            print(
                f"Early stopping at epoch {epoch}; best_epoch={best_epoch} best_val_mae={best_mae:.6f}",
                flush=True,
            )
            break

    train_ds.close()
    val_ds.close()
    print(f"Wrote outputs under {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
