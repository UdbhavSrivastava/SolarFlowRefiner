from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ERA_VARS = ["ssrd", "ssr", "ssrdc", "fdir", "cdir"]
DEFAULT_AUX_CHANNELS = ["vis047", "vis086", "ir133", "sza", "cm"]
ACCUMULATED_ERA_VARS = {
    "cdir",
    "fdir",
    "slhf",
    "sshf",
    "ssr",
    "ssrc",
    "ssrd",
    "ssrdc",
    "str",
    "strc",
    "strd",
    "strdc",
    "tisr",
    "tsr",
    "tsrc",
    "ttr",
    "ttrc",
    "uvb",
}
NONNEGATIVE_ERA_VARS = {
    "cdir",
    "fdir",
    "ssr",
    "ssrc",
    "ssrd",
    "ssrdc",
    "tisr",
    "tsr",
    "tsrc",
    "uvb",
}

QUARTERS = {
    "jan-mar": ("Jan-Mar.grib", "2018-01-01T00:00:00", "2018-04-01T00:00:00"),
    "apr-june": ("Apr-June.grib", "2018-04-01T00:00:00", "2018-07-01T00:00:00"),
    "july-sept": ("July-Sept.grib", "2018-07-01T00:00:00", "2018-10-01T00:00:00"),
    "oct-dec": ("Oct-Dec.grib", "2018-10-01T00:00:00", "2019-01-01T00:00:00"),
}


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class Progress:
    def __init__(self, verbose: bool, quiet: bool) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self.started = time.perf_counter()

    def log(self, message: str) -> None:
        if self.quiet:
            return
        elapsed = time.perf_counter() - self.started
        print(f"[{format_seconds(elapsed)}] {message}", flush=True)

    def debug(self, message: str) -> None:
        if self.verbose:
            self.log(message)

    def eta(self, completed: int, total: int) -> str:
        if completed <= 0 or total <= 0:
            return "unknown"
        elapsed = time.perf_counter() - self.started
        remaining = elapsed * (total - completed) / completed
        return format_seconds(remaining)

    def local_eta(self, started: float, completed: int, total: int) -> str:
        if completed <= 0 or total <= 0:
            return "unknown"
        elapsed = time.perf_counter() - started
        remaining = elapsed * (total - completed) / completed
        return format_seconds(remaining)


@dataclass
class RunningStats:
    count: np.ndarray
    sum: np.ndarray
    sumsq: np.ndarray
    min_value: np.ndarray
    max_value: np.ndarray

    @classmethod
    def create(cls, channels: int) -> "RunningStats":
        return cls(
            count=np.zeros(channels, dtype=np.float64),
            sum=np.zeros(channels, dtype=np.float64),
            sumsq=np.zeros(channels, dtype=np.float64),
            min_value=np.full(channels, np.inf, dtype=np.float64),
            max_value=np.full(channels, -np.inf, dtype=np.float64),
        )

    def update(self, array: np.ndarray) -> None:
        for channel in range(array.shape[1]):
            values = array[:, channel, :, :]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            self.count[channel] += finite.size
            self.sum[channel] += float(finite.sum(dtype=np.float64))
            self.sumsq[channel] += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))
            self.min_value[channel] = min(self.min_value[channel], float(finite.min()))
            self.max_value[channel] = max(self.max_value[channel], float(finite.max()))

    def to_dict(self, names: list[str]) -> dict[str, dict[str, float | int | None]]:
        output: dict[str, dict[str, float | int | None]] = {}
        for idx, name in enumerate(names):
            if self.count[idx] == 0:
                output[name] = {"count": 0, "mean": None, "std": None, "min": None, "max": None}
                continue
            mean = self.sum[idx] / self.count[idx]
            variance = max(self.sumsq[idx] / self.count[idx] - mean * mean, 0.0)
            output[name] = {
                "count": int(self.count[idx]),
                "mean": float(mean),
                "std": float(np.sqrt(variance)),
                "min": float(self.min_value[idx]),
                "max": float(self.max_value[idx]),
            }
        return output


class SolarCubeTileReader:
    def __init__(self, data_root: Path, tile_id: int, channels: list[str], progress: Progress) -> None:
        self.data_root = data_root
        self.tile_id = tile_id
        self.channels = channels
        self.progress = progress
        self.files: dict[str, h5py.File] = {}
        self.buffers: dict[str, io.BytesIO] = {}

    def __enter__(self) -> "SolarCubeTileReader":
        zip_path = self.data_root / "Solarcube" / f"SolarCube_2018_{self.tile_id}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        self.progress.log(f"Opening SolarCube tile {self.tile_id}: {zip_path.name}")
        started = time.perf_counter()
        with zipfile.ZipFile(zip_path) as archive:
            for idx, channel in enumerate(self.channels, start=1):
                member = f"SolarCube_2018_{self.tile_id}_{channel}.hdf"
                self.progress.debug(f"  SolarCube channel {idx}/{len(self.channels)}: reading {member}")
                channel_started = time.perf_counter()
                data = archive.read(member)
                buffer = io.BytesIO(data)
                self.buffers[channel] = buffer
                self.files[channel] = h5py.File(buffer, "r")
                shape = self.files[channel][channel].shape
                self.progress.debug(
                    f"  SolarCube channel {idx}/{len(self.channels)} loaded in "
                    f"{format_seconds(time.perf_counter() - channel_started)} shape={shape}"
                )
        self.progress.log(
            f"Opened SolarCube tile {self.tile_id} channels={len(self.channels)} "
            f"in {format_seconds(time.perf_counter() - started)}"
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for file in self.files.values():
            file.close()
        self.files.clear()
        self.buffers.clear()
        gc.collect()

    def read_hourly_mean(self, channel: str, hour_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.read_hourly_mean_from_frame_starts(channel, hour_indices.astype(np.int64) * 4)

    def read_hourly_mean_from_frame_starts(
        self, channel: str, frame_starts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dataset = self.files[channel][channel]
        self.progress.debug(
            f"  SolarCube {channel}: reading {len(frame_starts)} hourly samples "
            f"({len(frame_starts) * 4} fifteen-minute frames)"
        )
        started = time.perf_counter()
        frame_starts = frame_starts.astype(np.int64)
        frame_indices = np.concatenate([frame_starts + offset for offset in range(4)]).astype(np.int64)
        order = np.argsort(frame_indices)
        sorted_frames = frame_indices[order]
        arr = dataset[sorted_frames, :, :].astype(np.float32)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        arr = arr[inverse].reshape(4, len(frame_starts), arr.shape[-2], arr.shape[-1]).transpose(1, 0, 2, 3)

        fill = dataset.attrs.get("fillvalue")
        scale = dataset.attrs.get("scale_factor")
        if fill is not None:
            arr[arr == float(fill)] = np.nan
        if scale is not None:
            arr *= float(scale)

        valid_pct = np.isfinite(arr).mean(axis=(1, 2, 3)) * 100.0
        hourly = np.nanmean(arr, axis=1).astype(np.float32)
        self.progress.debug(
            f"  SolarCube {channel}: hourly mean ready in "
            f"{format_seconds(time.perf_counter() - started)}"
        )
        return hourly, valid_pct.astype(np.float32)

    def frame_count(self, channel: str) -> int:
        return int(self.files[channel][channel].shape[0])

    def read_frames(self, channel: str, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dataset = self.files[channel][channel]
        self.progress.debug(f"  SolarCube {channel}: reading {len(frame_indices)} exact frames")
        started = time.perf_counter()
        order = np.argsort(frame_indices)
        sorted_frames = frame_indices[order].astype(np.int64)
        arr = dataset[sorted_frames, :, :].astype(np.float32)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        arr = arr[inverse]

        fill = dataset.attrs.get("fillvalue")
        scale = dataset.attrs.get("scale_factor")
        if fill is not None:
            arr[arr == float(fill)] = np.nan
        if scale is not None:
            arr *= float(scale)

        valid_pct = np.isfinite(arr).mean(axis=(1, 2)) * 100.0
        self.progress.debug(
            f"  SolarCube {channel}: exact frames ready in "
            f"{format_seconds(time.perf_counter() - started)}"
        )
        return arr.astype(np.float32), valid_pct.astype(np.float32)


def parse_datetime(value: str) -> datetime:
    if "T" not in value:
        value = value + "T00:00:00"
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def hour_index(timestamp: datetime) -> int:
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    return int((timestamp - start).total_seconds() // 3600)


def hour_timestamp(index: int) -> datetime:
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    return start + timedelta(hours=int(index))


def solarcube_frame_starts(site: pd.Series, utc_hour_indices: np.ndarray, time_basis: str) -> np.ndarray:
    if time_basis == "utc":
        offset_hours = 0.0
    elif time_basis == "site_offset":
        offset_hours = float(site.get("time_offset", 0.0))
    else:
        raise ValueError(f"Unknown SolarCube time basis: {time_basis}")
    return np.rint((utc_hour_indices.astype(np.float64) + offset_hours) * 4.0).astype(np.int64)


def selected_quarters(values: list[str]) -> list[str]:
    if "all" in values:
        return ["jan-mar", "apr-june", "july-sept", "oct-dec"]
    for value in values:
        if value not in QUARTERS:
            raise ValueError(f"Unknown quarter {value!r}. Use one of {list(QUARTERS)} or all.")
    return values


def quarter_hours(name: str, start_date: str | None, end_date: str | None, hour_stride: int) -> np.ndarray:
    _grib, default_start, default_end = QUARTERS[name]
    start = parse_datetime(start_date) if start_date else parse_datetime(default_start)
    end = parse_datetime(end_date) if end_date else parse_datetime(default_end)
    q_start = parse_datetime(default_start)
    q_end = parse_datetime(default_end)
    start = max(start, q_start)
    end = min(end, q_end)
    if start >= end:
        return np.array([], dtype=np.int32)
    first = hour_index(start)
    last = hour_index(end)
    return np.arange(first, last, hour_stride, dtype=np.int32)


def read_sites(data_root: Path) -> pd.DataFrame:
    return pd.read_csv(data_root / "Solarcube" / "SolarCube_sitelist.csv")


def select_sites(sites: pd.DataFrame, split: str, tile_ids: list[int] | None) -> pd.DataFrame:
    selected = sites.copy()
    if split == "train":
        selected = selected[selected["test"].astype(str) == "0"]
    elif split == "test":
        selected = selected[selected["test"].astype(str) == "1"]
    elif split != "all":
        raise ValueError(f"Unknown split: {split}")
    if tile_ids:
        selected = selected[selected["tile_id"].isin(tile_ids)]
    if selected.empty:
        raise ValueError("No SolarCube sites selected")
    return selected.sort_values("tile_id")


def bicubic_resize_2d(arr: np.ndarray, target_size: int) -> np.ndarray:
    if np.isnan(arr).any():
        fill = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0
        source = np.nan_to_num(arr, nan=fill)
    else:
        source = arr
    image = Image.fromarray(source.astype(np.float32), mode="F")
    resized = image.resize((target_size, target_size), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32)


def convert_lons_for_era(ds: xr.Dataset, lon_min: float, lon_max: float) -> tuple[float, float]:
    era_lons = ds.longitude.values
    if np.nanmin(era_lons) >= 0:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    return lon_min, lon_max


def select_lat_lon_box(ds: xr.Dataset, field: xr.DataArray, site: pd.Series) -> xr.DataArray:
    lat_min = float(min(site["lat_ulcnr"], site["lat_lrcnr"]))
    lat_max = float(max(site["lat_ulcnr"], site["lat_lrcnr"]))
    lon_min = float(min(site["lon_ulcnr"], site["lon_lrcnr"]))
    lon_max = float(max(site["lon_ulcnr"], site["lon_lrcnr"]))
    lon_min, lon_max = convert_lons_for_era(ds, lon_min, lon_max)

    lat_values = ds.latitude.values
    lon_values = ds.longitude.values
    lat_slice = slice(lat_max, lat_min) if lat_values[0] > lat_values[-1] else slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max) if lon_values[0] < lon_values[-1] else slice(lon_max, lon_min)
    return field.sel(latitude=lat_slice, longitude=lon_slice)


def step_seconds(ds: xr.Dataset, step_index: int) -> int:
    value = ds.step.values[step_index]
    if np.issubdtype(type(value), np.timedelta64) or np.issubdtype(np.asarray(value).dtype, np.timedelta64):
        return int(np.asarray(value).astype("timedelta64[s]").astype(np.int64))
    return int(value)


def choose_era_index(
    ds: xr.Dataset,
    timestamp: datetime,
    site: pd.Series,
    primary_var: str,
    policy: str,
) -> tuple[dict[str, int], np.datetime64, float]:
    target = np.datetime64(timestamp.replace(tzinfo=None))
    valid_time = ds["valid_time"].values
    deltas = np.abs((valid_time - target).astype("timedelta64[s]").astype(np.int64))
    min_delta = int(np.nanmin(deltas))
    candidate_locs = np.argwhere(deltas == min_delta)
    if candidate_locs.shape[0] == 1 or policy == "first":
        loc = candidate_locs[0]
        indexers = {dim: int(index) for dim, index in zip(ds["valid_time"].dims, loc)}
        return indexers, valid_time[tuple(loc)], min_delta / 60.0

    if policy == "max_step":
        best_loc = max(candidate_locs, key=lambda loc: step_seconds(ds, int(loc[1])))
    elif policy == "primary_max":
        best_score = -np.inf
        best_loc = candidate_locs[0]
        for loc in candidate_locs:
            indexers = {dim: int(index) for dim, index in zip(ds["valid_time"].dims, loc)}
            try:
                field = ds[primary_var].isel(indexers)
                patch = select_lat_lon_box(ds, field, site).values.astype(np.float32)
                score = float(np.nanmean(patch))
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_loc = loc
    else:
        raise ValueError(f"Unknown ERA match policy: {policy}")

    indexers = {dim: int(index) for dim, index in zip(ds["valid_time"].dims, best_loc)}
    return indexers, valid_time[tuple(best_loc)], min_delta / 60.0


def era_step_interval_seconds(ds: xr.Dataset, step_index: int) -> int:
    current = step_seconds(ds, step_index)
    if step_index <= 0:
        return current if current > 0 else 3600
    previous = step_seconds(ds, step_index - 1)
    interval = current - previous
    return interval if interval > 0 else 3600


def extract_raw_era_patch(ds: xr.Dataset, site: pd.Series, variable: str, indexers: dict[str, int]) -> np.ndarray:
    field = ds[variable].isel(indexers)
    return select_lat_lon_box(ds, field, site).values.astype(np.float32)


def era_patch_shape(ds: xr.Dataset, site: pd.Series, variable: str, indexers: dict[str, int]) -> tuple[int, int]:
    field = ds[variable].isel(indexers)
    return tuple(select_lat_lon_box(ds, field, site).shape)


def convert_era_patch(
    ds: xr.Dataset,
    site: pd.Series,
    variable: str,
    indexers: dict[str, int],
    era_to_wm2: bool,
    accumulation_mode: str,
) -> np.ndarray:
    patch = extract_raw_era_patch(ds, site, variable, indexers)
    if not era_to_wm2 or variable not in ACCUMULATED_ERA_VARS:
        return patch

    if accumulation_mode == "divide":
        return patch / 3600.0
    if accumulation_mode == "none":
        return patch
    if accumulation_mode != "deaccumulate":
        raise ValueError(f"Unsupported ERA accumulation mode: {accumulation_mode}")

    step_index = int(indexers.get("step", -1))
    if "step" not in ds[variable].dims or step_index < 0:
        return patch / 3600.0

    interval_seconds = era_step_interval_seconds(ds, step_index)
    if step_index <= 0:
        return patch / float(interval_seconds)

    previous_indexers = dict(indexers)
    previous_indexers["step"] = step_index - 1
    previous_patch = extract_raw_era_patch(ds, site, variable, previous_indexers)
    return (patch - previous_patch) / float(interval_seconds)


def extract_era_for_hours(
    ds: xr.Dataset,
    site: pd.Series,
    hour_indices: np.ndarray,
    variables: list[str],
    target_size: int,
    era_to_wm2: bool,
    accumulation_mode: str,
    clip_nonnegative: bool,
    match_policy: str,
    primary_var: str,
    progress: Progress,
    progress_every: int,
    read_retries: int,
    retry_sleep: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    list[str],
]:
    upsampled_by_var: list[np.ndarray] = []
    coarse_by_var: list[list[np.ndarray]] = [[] for _ in variables]
    raw_coarse_by_var: list[list[np.ndarray]] = [[] for _ in variables]
    valid_times: list[str] = []
    delta_minutes: list[float] = []
    time_indices: list[int] = []
    step_indices: list[int] = []
    coarse_shapes: list[tuple[int, int]] = []
    read_ok = np.ones(len(hour_indices), dtype=bool)
    read_errors = ["" for _ in hour_indices]

    started = time.perf_counter()
    progress.log(f"  ERA matching {len(hour_indices)} hours with policy={match_policy}")
    index_records = []
    for idx, hidx in enumerate(hour_indices, start=1):
        timestamp = hour_timestamp(int(hidx))
        indexers, valid_time, delta_min = choose_era_index(ds, timestamp, site, primary_var, match_policy)
        index_records.append((indexers, valid_time, delta_min))
        valid_times.append(str(valid_time))
        delta_minutes.append(float(delta_min))
        time_indices.append(int(indexers.get("time", -1)))
        step_indices.append(int(indexers.get("step", -1)))
        if idx == 1 or idx == len(hour_indices) or idx % progress_every == 0:
            progress.log(
                f"  ERA matched {idx}/{len(hour_indices)} hours "
                f"ETA={progress.local_eta(started, idx, len(hour_indices))} "
                f"valid_time={valid_time} delta_min={delta_min:.1f}"
            )

    for var_idx, variable in enumerate(variables):
        if variable not in ds:
            raise KeyError(f"{variable!r} not in ERA dataset. Available variables: {list(ds.data_vars)}")
        var_started = time.perf_counter()
        progress.log(f"  ERA variable {variable}: extracting {len(hour_indices)} patches")
        frames = []
        for idx, (indexers, _valid_time, _delta_min) in enumerate(index_records, start=1):
            patch = None
            raw_patch = None
            last_error = ""
            for attempt in range(read_retries + 1):
                try:
                    raw_patch = extract_raw_era_patch(ds, site, variable, indexers)
                    patch = convert_era_patch(
                        ds,
                        site,
                        variable,
                        indexers,
                        era_to_wm2=era_to_wm2,
                        accumulation_mode=accumulation_mode,
                    )
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < read_retries:
                        progress.log(
                            f"    {variable}: read failed at {idx}/{len(index_records)} "
                            f"attempt {attempt + 1}/{read_retries + 1}; retrying: {last_error}"
                        )
                        gc.collect()
                        if retry_sleep > 0:
                            time.sleep(retry_sleep)
                    else:
                        read_ok[idx - 1] = False
                        if read_errors[idx - 1]:
                            read_errors[idx - 1] += " | "
                        read_errors[idx - 1] += f"{variable} {last_error}"
                        try:
                            shape = era_patch_shape(ds, site, variable, indexers)
                        except Exception:
                            shape = coarse_shapes[0] if coarse_shapes else (25, 25)
                        patch = np.full(shape, np.nan, dtype=np.float32)
                        raw_patch = np.full(shape, np.nan, dtype=np.float32)
                        progress.log(
                            f"    {variable}: read failed permanently at {idx}/{len(index_records)}; "
                            f"marking sample bad: {last_error}"
                        )
            if raw_patch is None:
                raw_patch = np.full_like(patch, np.nan, dtype=np.float32)
            if clip_nonnegative and variable in NONNEGATIVE_ERA_VARS:
                patch = np.clip(patch, 0.0, None)
            if not coarse_shapes:
                coarse_shapes.append(tuple(patch.shape))
            coarse_by_var[var_idx].append(patch)
            raw_coarse_by_var[var_idx].append(raw_patch)
            frames.append(bicubic_resize_2d(patch, target_size))
            if idx == 1 or idx == len(index_records) or idx % progress_every == 0:
                progress.log(
                    f"    {variable}: {idx}/{len(index_records)} patches "
                    f"ETA={progress.local_eta(var_started, idx, len(index_records))} "
                    f"patch_shape={patch.shape}"
                )
        upsampled_by_var.append(np.stack(frames).astype(np.float32))
        progress.log(
            f"  ERA variable {variable}: done in {format_seconds(time.perf_counter() - var_started)}"
        )

    era_up = np.stack(upsampled_by_var, axis=1).astype(np.float32)
    era_coarse = np.stack([np.stack(items) for items in coarse_by_var], axis=1).astype(np.float32)
    era_coarse_raw = np.stack([np.stack(items) for items in raw_coarse_by_var], axis=1).astype(np.float32)
    return (
        era_up,
        era_coarse,
        era_coarse_raw,
        valid_times,
        np.asarray(delta_minutes, dtype=np.float32),
        np.asarray(time_indices, dtype=np.int32),
        np.asarray(step_indices, dtype=np.int32),
        coarse_shapes,
        read_ok,
        read_errors,
    )


def write_index_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chunk",
                "row",
                "split",
                "quarter",
                "tile_id",
                "site_id",
                "site_name",
                "hour_index",
                "frame_start",
                "utc_time",
                "era_valid_time",
                "era_delta_minutes",
                "era_time_index",
                "era_step_index",
                "target_ssr_mean",
                "era_ssrd_mean",
                "era_ssr_mean",
                "cm_mean",
            ],
        )
        writer.writeheader()


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writerows(rows)


def bad_row(
    site: pd.Series,
    quarter: str,
    hour_idx: int,
    reason: str,
    target_mean: float,
    era_ssrd_mean: float | None,
    era_valid_time: str = "",
    era_delta_minutes: float | str = "",
    era_time_index: int | str = "",
    era_step_index: int | str = "",
    era_error: str = "",
) -> dict[str, Any]:
    return {
        "quarter": quarter,
        "tile_id": int(site["tile_id"]),
        "site_id": int(site["id"]),
        "site_name": site["name"],
        "hour_index": int(hour_idx),
        "frame_start": int(hour_idx) * 4,
        "utc_time": hour_timestamp(int(hour_idx)).isoformat(),
        "reason": reason,
        "target_ssr_mean": target_mean,
        "era_ssrd_mean": "" if era_ssrd_mean is None else era_ssrd_mean,
        "era_valid_time": era_valid_time,
        "era_delta_minutes": era_delta_minutes,
        "era_time_index": era_time_index,
        "era_step_index": era_step_index,
        "era_error": era_error,
    }


def write_bad_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "quarter",
                "tile_id",
                "site_id",
                "site_name",
                "hour_index",
                "frame_start",
                "utc_time",
                "reason",
                "target_ssr_mean",
                "era_ssrd_mean",
                "era_valid_time",
                "era_delta_minutes",
                "era_time_index",
                "era_step_index",
                "era_error",
            ],
        )
        writer.writeheader()


def save_chunk(
    out_dir: Path,
    split_name: str,
    tile_id: int,
    part_counter: int,
    x: np.ndarray,
    y: np.ndarray,
    y_exact: np.ndarray | None,
    valid_mask: np.ndarray,
    valid_mask_exact: np.ndarray | None,
    hour_indices: np.ndarray,
    frame_starts: np.ndarray,
    site_id: int,
    era_valid_times: list[str],
    era_time_indices: np.ndarray,
    era_step_indices: np.ndarray,
    save_era_coarse: bool,
    era_coarse: np.ndarray,
    era_coarse_raw: np.ndarray | None = None,
) -> Path:
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    path = chunks_dir / f"{split_name}_tile_{tile_id:02d}_part_{part_counter:05d}.npz"
    payload: dict[str, Any] = {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "valid_mask": valid_mask.astype(np.uint8),
        "hour_index": hour_indices.astype(np.int32),
        "frame_start": frame_starts.astype(np.int32),
        "tile_id": np.full(len(hour_indices), tile_id, dtype=np.int16),
        "site_id": np.full(len(hour_indices), site_id, dtype=np.int16),
        "era_valid_time": np.asarray(era_valid_times),
        "era_time_index": era_time_indices.astype(np.int32),
        "era_step_index": era_step_indices.astype(np.int32),
    }
    if y_exact is not None:
        payload["y_exact"] = y_exact.astype(np.float32)
    if valid_mask_exact is not None:
        payload["valid_mask_exact"] = valid_mask_exact.astype(np.uint8)
    if save_era_coarse:
        payload["era_coarse"] = era_coarse.astype(np.float32)
        if era_coarse_raw is not None:
            payload["era_coarse_raw"] = era_coarse_raw.astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly Task B ERA5-to-SolarCube dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--quarters", nargs="+", default=["jan-mar"], help="jan-mar apr-june july-sept oct-dec or all")
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
    parser.add_argument("--end-date", default=None, help="Optional exclusive end date")
    parser.add_argument("--hour-stride", type=int, default=1)
    parser.add_argument("--tile-ids", type=int, nargs="*", default=None)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--era-vars", nargs="+", default=DEFAULT_ERA_VARS)
    parser.add_argument("--aux-channels", nargs="+", default=DEFAULT_AUX_CHANNELS)
    parser.add_argument("--target-size", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-hours-per-tile", type=int, default=None)
    parser.add_argument("--min-target-ssr-mean", type=float, default=10.0)
    parser.add_argument("--min-valid-target-pct", type=float, default=98.0)
    parser.add_argument("--min-valid-visible-pct", type=float, default=80.0)
    parser.add_argument("--flag-target-mean", type=float, default=200.0)
    parser.add_argument("--flag-era-ssrd-mean", type=float, default=10.0)
    parser.add_argument(
        "--min-era-ssrd-mean",
        type=float,
        default=10.0,
        help="Drop daytime SolarCube samples when ERA ssrd mean is below this value after unit conversion.",
    )
    parser.add_argument("--include-night", action="store_true")
    parser.add_argument("--include-flagged-era-mismatch", action="store_true")
    parser.add_argument(
        "--solarcube-time-basis",
        choices=["utc", "site_offset"],
        default="utc",
        help=(
            "How UTC ERA hours map to SolarCube frame indices. "
            "utc uses frame_start=utc_hour*4. site_offset uses frame_start=(utc_hour+site.time_offset)*4."
        ),
    )
    parser.add_argument("--era-match-policy", choices=["max_step", "primary_max", "first"], default="primary_max")
    parser.add_argument("--primary-era-var", default="ssrd")
    parser.add_argument(
        "--era-accumulation-mode",
        choices=["deaccumulate", "divide", "none"],
        default="deaccumulate",
        help="How to convert accumulated ERA radiation/flux fields. deaccumulate uses current-minus-previous step.",
    )
    parser.add_argument("--no-era-to-wm2", dest="era_to_wm2", action="store_false")
    parser.add_argument("--no-clip-nonnegative-era", dest="clip_nonnegative_era", action="store_false")
    parser.add_argument("--era-read-retries", type=int, default=2)
    parser.add_argument("--era-retry-sleep", type=float, default=1.0)
    parser.add_argument("--save-era-coarse", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25, help="Print loop progress every N hours/patches.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(era_to_wm2=True, clip_nonnegative_era=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    progress = Progress(verbose=args.verbose, quiet=args.quiet)
    progress.log("Starting Task B hourly preprocessing")
    args.progress_every = max(1, int(args.progress_every))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "samples_index.csv"
    bad_path = args.out_dir / "bad_samples.csv"
    write_index_header(index_path)
    write_bad_header(bad_path)

    sites = select_sites(read_sites(args.data_root), args.split, args.tile_ids)
    quarters = selected_quarters(args.quarters)
    solarcube_channels = ["ssr"] + args.aux_channels
    x_channels = [f"era_{name}" for name in args.era_vars] + args.aux_channels
    y_channels = ["solarcube_ssr_hourly"]

    metadata = {
        "task": "Task B ERA5 to SolarCube hourly downscaling",
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "quarters": quarters,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "hour_stride": args.hour_stride,
        "split": args.split,
        "tile_ids": [int(value) for value in sites["tile_id"].tolist()],
        "era_vars": args.era_vars,
        "aux_channels": args.aux_channels,
        "x_channels": x_channels,
        "y_channels": y_channels,
        "target_size": args.target_size,
        "era_to_wm2": args.era_to_wm2,
        "era_accumulation_mode": args.era_accumulation_mode,
        "era_read_retries": args.era_read_retries,
        "era_retry_sleep": args.era_retry_sleep,
        "clip_nonnegative_era": args.clip_nonnegative_era,
        "era_match_policy": args.era_match_policy,
        "primary_era_var": args.primary_era_var,
        "include_night": args.include_night,
        "include_flagged_era_mismatch": args.include_flagged_era_mismatch,
        "solarcube_time_basis": args.solarcube_time_basis,
        "min_era_ssrd_mean": args.min_era_ssrd_mean,
        "target_definition": "hourly mean SolarCube ssr from four 15-minute frames",
        "input_definition": "ERA5 variables upsampled to 120x120 plus hourly SolarCube auxiliary channels",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    x_stats = RunningStats.create(len(x_channels))
    y_stats = RunningStats.create(len(y_channels))
    total_samples = 0
    total_bad = 0
    part_counter = 0
    total_site_quarters = len(sites) * len(quarters)
    completed_site_quarters = 0
    total_planned_chunks = 0
    total_planned_hours = 0
    for quarter_name in quarters:
        q_hours = quarter_hours(quarter_name, args.start_date, args.end_date, args.hour_stride)
        if args.max_hours_per_tile is not None:
            q_hours = q_hours[: args.max_hours_per_tile]
        total_planned_hours += len(q_hours) * len(sites)
        if len(q_hours):
            total_planned_chunks += int(np.ceil(len(q_hours) / args.chunk_size)) * len(sites)
    progress.log(
        f"Selected sites={len(sites)} quarters={len(quarters)} "
        f"planned_site_hours={total_planned_hours} planned_chunks={total_planned_chunks} "
        f"chunk_size={args.chunk_size} progress_every={args.progress_every}"
    )
    completed_chunks = 0
    completed_site_hours = 0

    for quarter in quarters:
        grib_name, _q_start, _q_end = QUARTERS[quarter]
        hour_candidates = quarter_hours(quarter, args.start_date, args.end_date, args.hour_stride)
        if len(hour_candidates) == 0:
            progress.log(f"Quarter {quarter}: no hours selected")
            continue

        era_path = args.data_root / "ERA" / grib_name
        progress.log(f"Quarter {quarter}: selected_hours={len(hour_candidates)}")
        progress.log(f"Opening ERA {quarter}: {era_path}")
        era_started = time.perf_counter()
        ds = xr.open_dataset(era_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        progress.log(
            f"Opened ERA {quarter} in {format_seconds(time.perf_counter() - era_started)}; "
            f"dims={dict(ds.sizes)} vars={list(ds.data_vars)}"
        )

        try:
            for _, site in sites.iterrows():
                completed_site_quarters += 1
                tile_id = int(site["tile_id"])
                site_id = int(site["id"])
                split_name = "test" if str(site["test"]) == "1" else "train"
                progress.log(
                    f"{quarter} site {completed_site_quarters}/{total_site_quarters}: "
                    f"tile={tile_id} site={site['name']} split={split_name} "
                    f"ETA={progress.eta(completed_site_quarters - 1, total_site_quarters)}"
                )

                tile_started = time.perf_counter()
                with SolarCubeTileReader(args.data_root, tile_id, solarcube_channels, progress) as reader:
                    selected_hours_for_tile = hour_candidates
                    if args.max_hours_per_tile is not None:
                        selected_hours_for_tile = selected_hours_for_tile[: args.max_hours_per_tile]
                    progress.log(
                        f"Tile {tile_id}: selected_hours={len(selected_hours_for_tile)} "
                        f"chunks={int(np.ceil(len(selected_hours_for_tile) / args.chunk_size))}"
                    )

                    for chunk_start in range(0, len(selected_hours_for_tile), args.chunk_size):
                        chunk_hours = selected_hours_for_tile[chunk_start : chunk_start + args.chunk_size]
                        solarcube_frame_starts_for_chunk = solarcube_frame_starts(
                            site, chunk_hours, args.solarcube_time_basis
                        )
                        frame_count = reader.frame_count("ssr")
                        frame_in_range = (solarcube_frame_starts_for_chunk >= 0) & (
                            solarcube_frame_starts_for_chunk + 3 < frame_count
                        )
                        if not np.all(frame_in_range):
                            bad_rows = [
                                bad_row(
                                    site,
                                    quarter,
                                    int(hidx),
                                    "filtered_solarcube_frame_out_of_range",
                                    float("nan"),
                                    None,
                                )
                                for hidx, ok in zip(chunk_hours, frame_in_range)
                                if not ok
                            ]
                            append_rows(bad_path, bad_rows)
                            total_bad += len(bad_rows)
                            chunk_hours = chunk_hours[frame_in_range]
                            solarcube_frame_starts_for_chunk = solarcube_frame_starts_for_chunk[frame_in_range]
                            if len(chunk_hours) == 0:
                                progress.log(f"Tile {tile_id}: all chunk hours outside SolarCube frame range")
                                completed_chunks += 1
                                completed_site_hours += len(frame_in_range)
                                continue
                        chunk_started = time.perf_counter()
                        progress.log(
                            f"Chunk {completed_chunks + 1}/{total_planned_chunks}: "
                            f"{quarter} tile={tile_id} hours {chunk_start + 1}-"
                            f"{chunk_start + len(chunk_hours)} of {len(selected_hours_for_tile)} "
                            f"overall_ETA={progress.eta(completed_chunks, total_planned_chunks)}"
                        )

                        solarcube_started = time.perf_counter()
                        progress.log("  SolarCube target: reading hourly ssr")
                        ssr, ssr_valid_pct = reader.read_hourly_mean_from_frame_starts(
                            "ssr", solarcube_frame_starts_for_chunk
                        )
                        progress.log("  SolarCube target: reading exact-hour ssr frame")
                        ssr_exact, _ssr_exact_valid_pct = reader.read_frames(
                            "ssr", solarcube_frame_starts_for_chunk
                        )
                        aux_data: dict[str, np.ndarray] = {}
                        aux_valid: dict[str, np.ndarray] = {}
                        for channel in args.aux_channels:
                            progress.log(f"  SolarCube aux: reading hourly {channel}")
                            aux_data[channel], aux_valid[channel] = reader.read_hourly_mean_from_frame_starts(
                                channel, solarcube_frame_starts_for_chunk
                            )
                        progress.log(
                            f"  SolarCube chunk read done in "
                            f"{format_seconds(time.perf_counter() - solarcube_started)}"
                        )

                        target_mean = np.nanmean(ssr, axis=(1, 2))
                        keep = ssr_valid_pct >= args.min_valid_target_pct
                        if not args.include_night:
                            keep &= target_mean >= args.min_target_ssr_mean
                            if "vis047" in aux_valid:
                                keep &= aux_valid["vis047"] >= args.min_valid_visible_pct
                            if "vis086" in aux_valid:
                                keep &= aux_valid["vis086"] >= args.min_valid_visible_pct
                        progress.log(
                            f"  QC SolarCube/daytime: kept={int(np.sum(keep))}/{len(chunk_hours)} "
                            f"filtered={len(chunk_hours) - int(np.sum(keep))} "
                            f"target_mean_range=({float(np.nanmin(target_mean)):.2f}, {float(np.nanmax(target_mean)):.2f})"
                        )

                        if not np.any(keep):
                            bad_rows = [
                                bad_row(site, quarter, int(hidx), "filtered_solarcube_daytime_or_validity", float(mean), None)
                                for hidx, mean in zip(chunk_hours, target_mean)
                            ]
                            append_rows(bad_path, bad_rows)
                            total_bad += len(bad_rows)
                            progress.log(f"Tile {tile_id}: no kept hours in this chunk")
                            completed_chunks += 1
                            completed_site_hours += len(chunk_hours)
                            progress.log(
                                f"Progress: chunks={completed_chunks}/{total_planned_chunks} "
                                f"site_hours_seen={completed_site_hours}/{total_planned_hours} "
                                f"overall_ETA={progress.eta(completed_chunks, total_planned_chunks)}"
                            )
                            continue

                        kept_hours = chunk_hours[keep]
                        kept_frame_starts = solarcube_frame_starts_for_chunk[keep]
                        ssr_kept = ssr[keep]
                        ssr_exact_kept = ssr_exact[keep]
                        aux_kept = {name: values[keep] for name, values in aux_data.items()}
                        target_mean_kept = target_mean[keep]
                        valid_mask = np.isfinite(ssr_kept)[:, None, :, :]
                        valid_mask_exact = np.isfinite(ssr_exact_kept)[:, None, :, :]
                        y = np.where(valid_mask, ssr_kept[:, None, :, :], 0.0).astype(np.float32)
                        y_exact = np.where(
                            valid_mask_exact, ssr_exact_kept[:, None, :, :], 0.0
                        ).astype(np.float32)

                        (
                            era_up,
                            era_coarse,
                            era_coarse_raw,
                            era_valid_times,
                            era_delta_min,
                            era_time_idx,
                            era_step_idx,
                            coarse_shapes,
                            era_read_ok,
                            era_read_errors,
                        ) = (
                            extract_era_for_hours(
                                ds,
                                site,
                                kept_hours,
                                args.era_vars,
                                target_size=args.target_size,
                                era_to_wm2=args.era_to_wm2,
                                accumulation_mode=args.era_accumulation_mode,
                                clip_nonnegative=args.clip_nonnegative_era,
                                match_policy=args.era_match_policy,
                                primary_var=args.primary_era_var,
                                progress=progress,
                                progress_every=args.progress_every,
                                read_retries=args.era_read_retries,
                                retry_sleep=args.era_retry_sleep,
                            )
                        )

                        era_ssrd_mean = None
                        if "ssrd" in args.era_vars:
                            ssrd_index = args.era_vars.index("ssrd")
                            era_ssrd_mean = np.nanmean(era_up[:, ssrd_index], axis=(1, 2))
                            mismatch = (~era_read_ok) | (era_ssrd_mean < args.min_era_ssrd_mean)
                            severe_mismatch = (target_mean_kept > args.flag_target_mean) & (
                                era_ssrd_mean < args.flag_era_ssrd_mean
                            )
                        else:
                            mismatch = ~era_read_ok
                            severe_mismatch = np.zeros(len(kept_hours), dtype=bool)

                        if np.any(mismatch):
                            bad_rows = [
                                bad_row(
                                    site,
                                    quarter,
                                    int(hidx),
                                    "era_read_failed"
                                    if not bool(era_read_ok[row_number])
                                    else (
                                        "era_ssrd_low_when_solarcube_high"
                                        if bool(severe_mismatch[row_number])
                                        else "era_ssrd_low_for_daytime_solarcube"
                                    ),
                                    float(tmean),
                                    float(emean) if era_ssrd_mean is not None else None,
                                    era_valid_time=era_valid_times[row_number],
                                    era_delta_minutes=float(era_delta_min[row_number]),
                                    era_time_index=int(era_time_idx[row_number]),
                                    era_step_index=int(era_step_idx[row_number]),
                                    era_error=era_read_errors[row_number],
                                )
                                for row_number, (hidx, tmean, emean, flag) in enumerate(
                                    zip(
                                        kept_hours,
                                        target_mean_kept,
                                        era_ssrd_mean if era_ssrd_mean is not None else np.full(len(kept_hours), np.nan),
                                        mismatch,
                                    )
                                )
                                if flag
                            ]
                            append_rows(bad_path, bad_rows)
                            total_bad += len(bad_rows)
                            progress.log(f"Tile {tile_id}: flagged {len(bad_rows)} ERA/SolarCube mismatch samples")

                        final_keep = np.ones(len(kept_hours), dtype=bool)
                        if not args.include_flagged_era_mismatch:
                            final_keep &= ~mismatch

                        if not np.any(final_keep):
                            progress.log(f"Tile {tile_id}: all kept hours were removed by ERA mismatch QC")
                            completed_chunks += 1
                            completed_site_hours += len(chunk_hours)
                            progress.log(
                                f"Progress: chunks={completed_chunks}/{total_planned_chunks} "
                                f"site_hours_seen={completed_site_hours}/{total_planned_hours} "
                                f"overall_ETA={progress.eta(completed_chunks, total_planned_chunks)}"
                            )
                            continue

                        kept_hours = kept_hours[final_keep]
                        kept_frame_starts = kept_frame_starts[final_keep]
                        ssr_kept = ssr_kept[final_keep]
                        ssr_exact_kept = ssr_exact_kept[final_keep]
                        target_mean_kept = target_mean_kept[final_keep]
                        y = y[final_keep]
                        y_exact = y_exact[final_keep]
                        valid_mask = valid_mask[final_keep]
                        valid_mask_exact = valid_mask_exact[final_keep]
                        era_up = era_up[final_keep]
                        era_coarse = era_coarse[final_keep]
                        era_coarse_raw = era_coarse_raw[final_keep]
                        era_valid_times = [value for value, flag in zip(era_valid_times, final_keep) if flag]
                        era_delta_min = era_delta_min[final_keep]
                        era_time_idx = era_time_idx[final_keep]
                        era_step_idx = era_step_idx[final_keep]
                        era_read_ok = era_read_ok[final_keep]
                        era_read_errors = [value for value, flag in zip(era_read_errors, final_keep) if flag]
                        aux_kept = {name: values[final_keep] for name, values in aux_kept.items()}
                        if era_ssrd_mean is not None:
                            era_ssrd_mean = era_ssrd_mean[final_keep]

                        x_parts = [np.nan_to_num(era_up, nan=0.0)]
                        for channel in args.aux_channels:
                            x_parts.append(np.nan_to_num(aux_kept[channel], nan=0.0)[:, None, :, :])
                        x = np.concatenate(x_parts, axis=1).astype(np.float32)

                        chunk_path = save_chunk(
                            args.out_dir,
                            split_name,
                            tile_id,
                            part_counter,
                            x,
                            y,
                            y_exact,
                            valid_mask,
                            valid_mask_exact,
                            kept_hours,
                            kept_frame_starts,
                            site_id,
                            era_valid_times,
                            era_time_idx,
                            era_step_idx,
                            save_era_coarse=args.save_era_coarse,
                            era_coarse=era_coarse,
                            era_coarse_raw=era_coarse_raw,
                        )
                        relative_chunk = str(chunk_path.relative_to(args.out_dir))

                        x_stats.update(x)
                        y_stats.update(y)

                        cm_mean = np.nanmean(aux_kept["cm"], axis=(1, 2)) if "cm" in aux_kept else np.full(len(kept_hours), np.nan)
                        era_ssr_mean = (
                            np.nanmean(era_up[:, args.era_vars.index("ssr")], axis=(1, 2))
                            if "ssr" in args.era_vars
                            else np.full(len(kept_hours), np.nan)
                        )
                        era_ssrd_index_mean = (
                            era_ssrd_mean if era_ssrd_mean is not None else np.full(len(kept_hours), np.nan)
                        )
                        index_rows = []
                        for row_idx, hidx in enumerate(kept_hours):
                            index_rows.append(
                                {
                                    "chunk": relative_chunk,
                                    "row": row_idx,
                                    "split": split_name,
                                    "quarter": quarter,
                                    "tile_id": tile_id,
                                    "site_id": site_id,
                                    "site_name": site["name"],
                                    "hour_index": int(hidx),
                                    "frame_start": int(kept_frame_starts[row_idx]),
                                    "utc_time": hour_timestamp(int(hidx)).isoformat(),
                                    "era_valid_time": era_valid_times[row_idx],
                                    "era_delta_minutes": float(era_delta_min[row_idx]),
                                    "era_time_index": int(era_time_idx[row_idx]),
                                    "era_step_index": int(era_step_idx[row_idx]),
                                    "target_ssr_mean": float(target_mean_kept[row_idx]),
                                    "era_ssrd_mean": float(era_ssrd_index_mean[row_idx]),
                                    "era_ssr_mean": float(era_ssr_mean[row_idx]),
                                    "cm_mean": float(cm_mean[row_idx]),
                                }
                            )
                        append_rows(index_path, index_rows)

                        total_samples += len(kept_hours)
                        part_counter += 1
                        progress.log(
                            f"Tile {tile_id}: wrote {relative_chunk} rows={len(kept_hours)} "
                            f"total_samples={total_samples} chunk_time={format_seconds(time.perf_counter() - chunk_started)}"
                        )
                        completed_chunks += 1
                        completed_site_hours += len(chunk_hours)
                        progress.log(
                            f"Progress: chunks={completed_chunks}/{total_planned_chunks} "
                            f"site_hours_seen={completed_site_hours}/{total_planned_hours} "
                            f"overall_ETA={progress.eta(completed_chunks, total_planned_chunks)}"
                        )

                        del x, y, y_exact, valid_mask, valid_mask_exact
                        del era_up, era_coarse, era_coarse_raw, aux_kept, ssr_kept, ssr_exact_kept
                        gc.collect()

                progress.log(f"Finished tile {tile_id} in {format_seconds(time.perf_counter() - tile_started)}")
        finally:
            ds.close()
            gc.collect()

    stats = {
        "total_samples": total_samples,
        "total_bad_or_flagged_samples": total_bad,
        "x": x_stats.to_dict(x_channels),
        "y": y_stats.to_dict(y_channels),
    }
    (args.out_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    progress.log(f"Done. total_samples={total_samples} total_bad_or_flagged={total_bad}")
    progress.log(f"Output directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
