"""
frequency_map.py

Memory-safe + multiprocessing Allen Neuropixels LFP natural-scenes
bandpower featurization.

Pipeline
--------
1. Load ViT logits and create image-level animate/inanimate labels.
2. Load Allen ecephys session and one probe's LFP handle.
3. Select only natural_scenes stimulus presentations.
4. Robustly convert the stimulus 'frame' column to numeric.
5. Explicitly remove blank/empty presentations where frame == -1.
6. Read actual stimulus timing from the Allen presentation table:
      start_time + stop_time
   or:
      start_time + duration
7. Build a disk-backed raw padded Page tensor:

      raw_tensor[presentation, padded_time, channel]

   This contains only natural_scenes presentations, not the whole LFP recording.

8. Use multiprocessing to process each presentation:
      raw padded segment
      -> fill NaNs
      -> notch / optional filters
      -> crop baseline and stimulus windows
      -> Welch bandpower
      -> log(P_stim + eps) - log(P_baseline + eps)

9. Save final arrays:

      X_power: shape (n_presentations, n_channels, n_bands)
      X_flat:  shape (n_presentations, n_channels * n_bands)
      y:       shape (n_presentations,)

Recommended decoding
--------------------
Use grouped CV with groups=image_indices, so repeated presentations of the same
image do not leak across train/test.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
    EcephysProjectCache,
)


# =============================================================================
# Config
# =============================================================================

MANIFEST_PATH = "/media/maria/notsudata/AllenNeuropixels/manifest.json"

# IMPORTANT:
# This must be the .npz file containing key "natural_scenes", not a .py file.
VIT_LOGITS_PATH = "/home/maria/ProjectionSort/data/google_vit-base-patch16-224_embeddings_logits.pkl"

SESSION_ID = 739448407
CHOOSE_PROBE = "probeA"

OUT_DIR = Path(f"/media/maria/notsudata/NeuropixelDataProc/lfp_natural_scenes_tensor_{CHOOSE_PROBE}_session{SESSION_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_MEMMAP_PATH = OUT_DIR / "raw_padded_lfp_tensor.float32.memmap"
X_POWER_MEMMAP_PATH = OUT_DIR / "X_power.float32.memmap"
STIM_POWER_MEMMAP_PATH = OUT_DIR / "stim_power.float32.memmap"
BASELINE_POWER_MEMMAP_PATH = OUT_DIR / "baseline_power.float32.memmap"
FINAL_NPZ_PATH = OUT_DIR / "lfp_natural_scenes_bandpower_features.npz"

# Baseline window relative to stimulus onset.
BASELINE_START = -0.25
BASELINE_END = 0.0

# Padding around baseline/stimulus before filtering.
# This reduces filter edge artifacts.
FILTER_PAD_SEC = 1.0

# Allen ecephys data are US-collected, usually 60 Hz line noise.
# Set to None to disable notch filtering.
LINE_FREQ_HZ: Optional[float] = 60.0
MAX_NOTCH_HZ = 500.0

# For per-presentation filtering, a very low highpass can introduce edge weirdness.
# I recommend starting with None and relying on per-window detrending for bandpower.
# You can set this to 0.5 later if needed.
HIGHPASS_HZ: Optional[float] = None

# Optional lowpass. Usually unnecessary at native Allen LFP sampling rate.
LOWPASS_HZ: Optional[float] = None

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 80.0),
    "mid_gamma": (80.0, 150.0),
    "high_gamma": (150.0, 250.0),
}

EPS = 1e-12

# Start conservative. Increase after it works.
N_WORKERS = max(1, min(4, os.cpu_count() or 1))
CHUNK_SIZE = 32

# Delete old outputs before running.
OVERWRITE = True


# =============================================================================
# ViT labels
# =============================================================================

def load_vit_animate_labels(vit_logits_path: str) -> np.ndarray:
    """
    Load ViT logits and derive animate/inanimate labels.

    Expected file:
        .npz with key "natural_scenes"

    Expected array:
        shape (n_images, 1000)

    Label rule:
        top1 <= 397 -> animate
        top1 > 397  -> inanimate

    Returns
    -------
    image_labels:
        shape (n_images,)
        0 = inanimate
        1 = animate
    """
    vit_file = np.load(vit_logits_path, allow_pickle=True)

    if "natural_scenes" not in vit_file:
        raise KeyError(
            f"Expected key 'natural_scenes' in {vit_logits_path}. "
            f"Found keys: {list(vit_file.keys())}"
        )

    vit = vit_file["natural_scenes"]

    if vit.ndim != 2 or vit.shape[1] != 1000:
        raise ValueError(
            f"Expected ViT logits with shape (n_images, 1000), got {vit.shape}"
        )

    top1 = np.argmax(vit, axis=1)
    image_labels = (top1 <= 397).astype(np.int64)

    print(f"Loaded ViT logits: {vit.shape}")
    print(f"Image-level label counts [inanimate, animate]: {np.bincount(image_labels)}")
    print("Convention: 0 = inanimate, 1 = animate")

    return image_labels


# =============================================================================
# LFP utilities
# =============================================================================

def infer_lfp_sampling_rate(session, probe_id: int, lfp: xr.DataArray) -> float:
    """
    Prefer session.probes['lfp_sampling_rate']; fall back to median LFP time step.
    """
    if "lfp_sampling_rate" in session.probes.columns:
        fs = float(session.probes.loc[probe_id, "lfp_sampling_rate"])
        if np.isfinite(fs) and fs > 0:
            return fs

    times = np.asarray(lfp["time"].values, dtype=np.float64)
    dt = np.median(np.diff(times))
    fs = 1.0 / dt

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Could not infer LFP sampling rate")

    return float(fs)


def get_lfp_time_and_channel_info(
    lfp: xr.DataArray,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """
    Get dimension names and coordinates without loading the full LFP values.
    """
    if "time" not in lfp.dims:
        raise ValueError(f"Expected LFP to have 'time' dimension. Got dims={lfp.dims}")

    non_time_dims = [d for d in lfp.dims if d != "time"]
    if len(non_time_dims) != 1:
        raise ValueError(f"Expected exactly one non-time dimension. Got dims={lfp.dims}")

    time_dim = "time"
    channel_dim = non_time_dims[0]

    times = np.asarray(lfp[time_dim].values, dtype=np.float64)
    channel_ids = np.asarray(lfp[channel_dim].values)

    return time_dim, channel_dim, times, channel_ids


def load_lfp_segment(
    lfp: xr.DataArray,
    time_dim: str,
    channel_dim: str,
    i0: int,
    i1: int,
) -> np.ndarray:
    """
    Load one small LFP segment from xarray.

    Returns
    -------
    segment:
        shape (n_time, n_channels)
    """
    segment_da = lfp.isel({time_dim: slice(i0, i1)})
    segment_da = segment_da.transpose(time_dim, channel_dim)

    segment = np.asarray(segment_da.values, dtype=np.float32)

    if segment.ndim != 2:
        raise ValueError(f"Expected 2D LFP segment, got shape {segment.shape}")

    return segment


# =============================================================================
# Stimulus table
# =============================================================================

def get_natural_scenes_presentations(session) -> pd.DataFrame:
    """
    Get natural-scenes presentations and explicitly remove frame == -1 blanks.

    This function avoids np.isfinite directly on the 'frame' Series, because
    Allen tables may expose it as object/string/mixed dtype.
    """
    presentations = session.stimulus_presentations[
        session.stimulus_presentations.stimulus_name == "natural_scenes"
    ].copy()

    if len(presentations) == 0:
        raise ValueError("No natural_scenes presentations found")

    for col in ["start_time", "frame"]:
        if col not in presentations.columns:
            raise ValueError(
                f"Expected column '{col}'. Available columns: {list(presentations.columns)}"
            )

    before = len(presentations)

    presentations["start_time"] = pd.to_numeric(
        presentations["start_time"],
        errors="coerce",
    )

    presentations["frame"] = pd.to_numeric(
        presentations["frame"],
        errors="coerce",
    )

    presentations = presentations.dropna(subset=["start_time", "frame"]).copy()
    presentations["frame"] = presentations["frame"].astype(int)

    n_blank = int((presentations["frame"] == -1).sum())
    if n_blank:
        print(f"Filtering out {n_blank} blank presentations with frame == -1")

    n_other_negative = int((presentations["frame"] < -1).sum())
    if n_other_negative:
        print(f"Filtering out {n_other_negative} presentations with frame < -1")

    presentations = presentations[presentations["frame"] >= 0].copy()

    after = len(presentations)

    if after == 0:
        raise ValueError("No valid natural-scenes image presentations left")

    print(f"Natural-scenes presentations before filtering: {before}")
    print(f"Natural-scenes presentations after filtering:  {after}")
    print(
        f"Frame range after filtering: "
        f"{presentations['frame'].min()} to {presentations['frame'].max()}"
    )

    return presentations


def add_stimulus_end_times(presentations: pd.DataFrame) -> pd.DataFrame:
    """
    Add clean stimulus start/end/duration columns from the Allen table.

    Uses:
        stop_time if present
    otherwise:
        start_time + duration
    """
    presentations = presentations.copy()

    presentations["start_time"] = pd.to_numeric(
        presentations["start_time"],
        errors="coerce",
    )

    starts = presentations["start_time"].to_numpy(dtype=float)

    if "stop_time" in presentations.columns:
        presentations["stop_time"] = pd.to_numeric(
            presentations["stop_time"],
            errors="coerce",
        )
        ends = presentations["stop_time"].to_numpy(dtype=float)

    elif "duration" in presentations.columns:
        presentations["duration"] = pd.to_numeric(
            presentations["duration"],
            errors="coerce",
        )
        ends = starts + presentations["duration"].to_numpy(dtype=float)

    else:
        raise ValueError(
            "Presentation table must contain either 'stop_time' or 'duration'. "
            f"Available columns: {list(presentations.columns)}"
        )

    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)

    if (~valid).sum() > 0:
        print(f"Dropping {(~valid).sum()} presentations with invalid timing")

    presentations = presentations.iloc[np.where(valid)[0]].copy()

    presentations["_stim_start_time"] = starts[valid]
    presentations["_stim_end_time"] = ends[valid]
    presentations["_stim_duration"] = (
        presentations["_stim_end_time"] - presentations["_stim_start_time"]
    )

    print(
        "Stimulus durations from table: "
        f"min={presentations['_stim_duration'].min():.6f}s, "
        f"median={presentations['_stim_duration'].median():.6f}s, "
        f"max={presentations['_stim_duration'].max():.6f}s"
    )

    return presentations


def choose_common_stimulus_duration(
    presentations: pd.DataFrame,
    fs: float,
) -> tuple[float, int]:
    """
    Use the shortest actual natural-scenes presentation duration so every
    stimulus window has equal length and fits inside the true stimulus interval.
    """
    durations = presentations["_stim_duration"].to_numpy(dtype=float)

    common_duration = float(np.min(durations))
    n_samples = int(np.floor(common_duration * fs))

    if n_samples <= 1:
        raise ValueError(
            f"Common stimulus duration too short: {common_duration}s -> {n_samples} samples"
        )

    common_duration = n_samples / fs

    print(f"Using common stimulus duration: {common_duration:.6f}s")
    print(f"Stimulus samples per presentation: {n_samples}")

    return common_duration, n_samples


def compute_padded_window_geometry(
    fs: float,
    common_stim_duration: float,
) -> dict:
    """
    Define one common padded window relative to stimulus onset.

    Raw tensor stores:
        [padded_start_rel, padded_end_rel]

    relative to stimulus onset.

    Later, each worker crops:
        baseline window
        stimulus window
    from the filtered padded segment.
    """
    baseline_duration = BASELINE_END - BASELINE_START

    if baseline_duration <= 0:
        raise ValueError("Invalid baseline window")

    padded_start_rel = min(BASELINE_START, 0.0) - FILTER_PAD_SEC
    padded_end_rel = max(BASELINE_END, common_stim_duration) + FILTER_PAD_SEC

    padded_duration = padded_end_rel - padded_start_rel
    padded_n_samples = int(np.ceil(padded_duration * fs))

    if padded_n_samples <= 1:
        raise ValueError("Padded segment too short")

    # Make padded end sample-exact.
    padded_duration = padded_n_samples / fs
    padded_end_rel = padded_start_rel + padded_duration

    baseline_start_idx = int(round((BASELINE_START - padded_start_rel) * fs))
    baseline_n_samples = int(np.floor(baseline_duration * fs))

    stim_start_idx = int(round((0.0 - padded_start_rel) * fs))
    stim_n_samples = int(np.floor(common_stim_duration * fs))

    if baseline_start_idx < 0:
        raise ValueError("Computed negative baseline_start_idx")

    if stim_start_idx < 0:
        raise ValueError("Computed negative stim_start_idx")

    if baseline_start_idx + baseline_n_samples > padded_n_samples:
        raise ValueError("Baseline window exceeds padded segment")

    if stim_start_idx + stim_n_samples > padded_n_samples:
        raise ValueError("Stimulus window exceeds padded segment")

    geom = {
        "_fs": float(fs),
        "padded_start_rel": float(padded_start_rel),
        "padded_end_rel": float(padded_end_rel),
        "padded_duration": float(padded_duration),
        "padded_n_samples": int(padded_n_samples),
        "baseline_start_idx": int(baseline_start_idx),
        "baseline_n_samples": int(baseline_n_samples),
        "stim_start_idx": int(stim_start_idx),
        "stim_n_samples": int(stim_n_samples),
    }

    print("Window geometry:")
    for k, v in geom.items():
        print(f"  {k}: {v}")

    return geom


def filter_valid_presentations_for_padded_tensor(
    presentations: pd.DataFrame,
    times: np.ndarray,
    geom: dict,
) -> pd.DataFrame:
    """
    Keep presentations whose padded window fits inside the LFP recording and
    whose chosen stimulus window fits inside the true stimulus interval.
    """
    t_min = float(times[0])
    t_max = float(times[-1])

    fs = float(geom["_fs"])

    starts = presentations["_stim_start_time"].to_numpy(dtype=float)
    ends = presentations["_stim_end_time"].to_numpy(dtype=float)

    padded_starts = starts + geom["padded_start_rel"]
    padded_ends = starts + geom["padded_end_rel"]

    stim_duration = geom["stim_n_samples"] / fs
    stim_ends = starts + stim_duration

    valid = (
        np.isfinite(padded_starts)
        & np.isfinite(padded_ends)
        & np.isfinite(stim_ends)
        & (padded_starts >= t_min)
        & (padded_ends <= t_max)
        & (stim_ends <= ends)
    )

    print(f"Presentations valid for padded tensor: {valid.sum()} / {len(valid)}")

    if valid.sum() == 0:
        raise ValueError("No presentations left after padded-window filtering")

    return presentations.iloc[np.where(valid)[0]].copy()


# =============================================================================
# Disk-backed raw tensor construction
# =============================================================================

def remove_existing_outputs() -> None:
    paths = [
        RAW_MEMMAP_PATH,
        X_POWER_MEMMAP_PATH,
        STIM_POWER_MEMMAP_PATH,
        BASELINE_POWER_MEMMAP_PATH,
        FINAL_NPZ_PATH,
    ]

    for path in paths:
        if path.exists():
            print(f"Removing old output: {path}")
            path.unlink()


def build_raw_padded_lfp_tensor(
    lfp: xr.DataArray,
    time_dim: str,
    channel_dim: str,
    times: np.ndarray,
    presentations: pd.DataFrame,
    geom: dict,
    n_channels: int,
) -> np.memmap:
    """
    Build disk-backed raw tensor:

        raw[presentation, padded_time, channel]
    """
    n_presentations = len(presentations)
    padded_n_samples = int(geom["padded_n_samples"])

    raw = np.memmap(
        RAW_MEMMAP_PATH,
        dtype="float32",
        mode="w+",
        shape=(n_presentations, padded_n_samples, n_channels),
    )

    starts = presentations["_stim_start_time"].to_numpy(dtype=float)

    print("\nBuilding raw padded LFP Page tensor")
    print(f"Raw tensor shape: {raw.shape}")
    print(f"Raw tensor path:  {RAW_MEMMAP_PATH}")

    for i, stim_start in enumerate(starts):
        padded_start_abs = stim_start + geom["padded_start_rel"]

        i0 = int(np.searchsorted(times, padded_start_abs, side="left"))
        i1 = i0 + padded_n_samples

        segment = load_lfp_segment(
            lfp=lfp,
            time_dim=time_dim,
            channel_dim=channel_dim,
            i0=i0,
            i1=i1,
        )

        if segment.shape != (padded_n_samples, n_channels):
            raise RuntimeError(
                f"Segment shape mismatch at presentation {i}: "
                f"got {segment.shape}, expected {(padded_n_samples, n_channels)}"
            )

        raw[i, :, :] = segment

        if (i + 1) % 100 == 0 or (i + 1) == n_presentations:
            print(f"Loaded raw tensor segment {i + 1} / {n_presentations}")

    raw.flush()
    return raw


# =============================================================================
# Signal processing helpers
# =============================================================================

def fill_segment_nans(segment: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by interpolation per channel.
    """
    if np.isfinite(segment).all():
        return segment

    out = segment.copy()
    n_time, n_channels = out.shape
    x = np.arange(n_time)

    for ch in range(n_channels):
        y = out[:, ch]
        good = np.isfinite(y)

        if good.all():
            continue

        if good.sum() < 2:
            out[:, ch] = 0.0
        else:
            out[:, ch] = np.interp(x, x[good], y[good])

    return out


def apply_notch_filters_to_segment(
    segment: np.ndarray,
    fs: float,
    line_freq_hz: Optional[float],
    max_notch_hz: float,
    q: float = 30.0,
) -> np.ndarray:
    """
    Apply zero-phase notch filters to one padded presentation segment.
    """
    if line_freq_hz is None:
        return segment

    nyquist = fs / 2.0
    max_freq = min(max_notch_hz, nyquist * 0.95)

    out = segment

    k = 1
    while k * line_freq_hz <= max_freq:
        f0 = k * line_freq_hz
        b, a = signal.iirnotch(w0=f0, Q=q, fs=fs)

        padlen = 3 * max(len(a), len(b))
        if out.shape[0] > padlen:
            out = signal.filtfilt(b, a, out, axis=0)

        k += 1

    return out


def apply_optional_filters_to_segment(
    segment: np.ndarray,
    fs: float,
    highpass_hz: Optional[float],
    lowpass_hz: Optional[float],
) -> np.ndarray:
    """
    Apply optional highpass / lowpass filters to one padded segment.
    """
    out = segment
    nyquist = fs / 2.0

    if highpass_hz is not None:
        if not (0 < highpass_hz < nyquist):
            raise ValueError(f"Invalid highpass_hz={highpass_hz}")

        sos = signal.butter(
            4,
            highpass_hz,
            btype="highpass",
            fs=fs,
            output="sos",
        )

        out = signal.sosfiltfilt(sos, out, axis=0)

    if lowpass_hz is not None:
        if not (0 < lowpass_hz < nyquist):
            raise ValueError(f"Invalid lowpass_hz={lowpass_hz}")

        sos = signal.butter(
            6,
            lowpass_hz,
            btype="lowpass",
            fs=fs,
            output="sos",
        )

        out = signal.sosfiltfilt(sos, out, axis=0)

    return out


def preprocess_segment(segment: np.ndarray, fs: float) -> np.ndarray:
    """
    Fill NaNs and apply filters to one padded segment.
    """
    segment = fill_segment_nans(segment)

    segment = apply_notch_filters_to_segment(
        segment=segment,
        fs=fs,
        line_freq_hz=LINE_FREQ_HZ,
        max_notch_hz=MAX_NOTCH_HZ,
        q=30.0,
    )

    segment = apply_optional_filters_to_segment(
        segment=segment,
        fs=fs,
        highpass_hz=HIGHPASS_HZ,
        lowpass_hz=LOWPASS_HZ,
    )

    return segment


def compute_bandpower_for_window(
    window: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch bandpower for one window.

    Parameters
    ----------
    window:
        shape (n_time, n_channels)

    Returns
    -------
    power:
        shape (n_channels, n_bands)

    freqs:
        Welch frequency bins
    """
    if window.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {window.shape}")

    n_time, n_channels = window.shape

    window = signal.detrend(window, axis=0, type="constant")

    # Use full window to maximize frequency resolution for short stimuli.
    nperseg = n_time
    noverlap = 0

    freqs, psd = signal.welch(
        window,
        fs=fs,
        axis=0,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        average="mean",
    )

    band_names = list(bands.keys())
    power = np.zeros((n_channels, len(band_names)), dtype=np.float32)

    nyquist = fs / 2.0
    df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else fs / n_time

    for b_idx, band_name in enumerate(band_names):
        low, high = bands[band_name]

        if high > nyquist:
            raise ValueError(
                f"Band {band_name} high cutoff {high} exceeds Nyquist {nyquist}"
            )

        mask = (freqs >= low) & (freqs <= high)

        if mask.sum() == 0:
            power[:, b_idx] = np.nan
        elif mask.sum() == 1:
            power[:, b_idx] = psd[mask, :].squeeze(axis=0) * df
        else:
            power[:, b_idx] = np.trapz(psd[mask, :], freqs[mask], axis=0)

    return power, freqs


# =============================================================================
# Multiprocessing worker state
# =============================================================================

_WORKER_RAW = None
_WORKER_X = None
_WORKER_STIM = None
_WORKER_BASELINE = None
_WORKER_FS = None
_WORKER_GEOM = None


def init_worker(
    raw_path: str,
    x_path: str,
    stim_path: str,
    baseline_path: str,
    shape_raw: tuple[int, int, int],
    shape_x: tuple[int, int, int],
    fs: float,
    geom: dict,
) -> None:
    """
    Open memmaps inside each worker process.
    """
    global _WORKER_RAW
    global _WORKER_X
    global _WORKER_STIM
    global _WORKER_BASELINE
    global _WORKER_FS
    global _WORKER_GEOM

    _WORKER_FS = float(fs)
    _WORKER_GEOM = geom

    _WORKER_RAW = np.memmap(
        raw_path,
        dtype="float32",
        mode="r",
        shape=shape_raw,
    )

    _WORKER_X = np.memmap(
        x_path,
        dtype="float32",
        mode="r+",
        shape=shape_x,
    )

    _WORKER_STIM = np.memmap(
        stim_path,
        dtype="float32",
        mode="r+",
        shape=shape_x,
    )

    _WORKER_BASELINE = np.memmap(
        baseline_path,
        dtype="float32",
        mode="r+",
        shape=shape_x,
    )


def process_index_range(index_range: tuple[int, int]) -> tuple[int, int]:
    """
    Worker task: process presentations [start, stop).
    """
    start, stop = index_range

    baseline_start_idx = int(_WORKER_GEOM["baseline_start_idx"])
    baseline_n_samples = int(_WORKER_GEOM["baseline_n_samples"])

    stim_start_idx = int(_WORKER_GEOM["stim_start_idx"])
    stim_n_samples = int(_WORKER_GEOM["stim_n_samples"])

    for i in range(start, stop):
        segment = np.asarray(_WORKER_RAW[i, :, :], dtype=np.float64)

        segment = preprocess_segment(segment, fs=_WORKER_FS)

        baseline_window = segment[
            baseline_start_idx : baseline_start_idx + baseline_n_samples,
            :,
        ]

        stim_window = segment[
            stim_start_idx : stim_start_idx + stim_n_samples,
            :,
        ]

        if baseline_window.shape[0] != baseline_n_samples:
            raise RuntimeError(
                f"Baseline window mismatch for presentation {i}: "
                f"{baseline_window.shape[0]} vs {baseline_n_samples}"
            )

        if stim_window.shape[0] != stim_n_samples:
            raise RuntimeError(
                f"Stimulus window mismatch for presentation {i}: "
                f"{stim_window.shape[0]} vs {stim_n_samples}"
            )

        stim_power, _ = compute_bandpower_for_window(
            stim_window,
            fs=_WORKER_FS,
            bands=BANDS,
        )

        baseline_power, _ = compute_bandpower_for_window(
            baseline_window,
            fs=_WORKER_FS,
            bands=BANDS,
        )

        corrected = np.log(stim_power + EPS) - np.log(baseline_power + EPS)

        corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        stim_power = np.nan_to_num(stim_power, nan=0.0, posinf=0.0, neginf=0.0)
        baseline_power = np.nan_to_num(
            baseline_power,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        _WORKER_X[i, :, :] = corrected.astype(np.float32)
        _WORKER_STIM[i, :, :] = stim_power.astype(np.float32)
        _WORKER_BASELINE[i, :, :] = baseline_power.astype(np.float32)

    _WORKER_X.flush()
    _WORKER_STIM.flush()
    _WORKER_BASELINE.flush()

    return start, stop


def make_chunks(n: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def run_parallel_signal_processing(
    shape_raw: tuple[int, int, int],
    fs: float,
    geom: dict,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    """
    Run multiprocessing over the disk-backed raw Page tensor.
    """
    n_presentations, _, n_channels = shape_raw
    n_bands = len(BANDS)

    shape_x = (n_presentations, n_channels, n_bands)

    X_power = np.memmap(
        X_POWER_MEMMAP_PATH,
        dtype="float32",
        mode="w+",
        shape=shape_x,
    )

    stim_power = np.memmap(
        STIM_POWER_MEMMAP_PATH,
        dtype="float32",
        mode="w+",
        shape=shape_x,
    )

    baseline_power = np.memmap(
        BASELINE_POWER_MEMMAP_PATH,
        dtype="float32",
        mode="w+",
        shape=shape_x,
    )

    X_power[:] = 0.0
    stim_power[:] = 0.0
    baseline_power[:] = 0.0

    X_power.flush()
    stim_power.flush()
    baseline_power.flush()

    chunks = make_chunks(n_presentations, CHUNK_SIZE)

    print("\nRunning multiprocessing signal processing")
    print(f"Workers: {N_WORKERS}")
    print(f"Chunks:  {len(chunks)}")
    print(f"Output shape: {shape_x}")

    with mp.Pool(
        processes=N_WORKERS,
        initializer=init_worker,
        initargs=(
            str(RAW_MEMMAP_PATH),
            str(X_POWER_MEMMAP_PATH),
            str(STIM_POWER_MEMMAP_PATH),
            str(BASELINE_POWER_MEMMAP_PATH),
            shape_raw,
            shape_x,
            fs,
            geom,
        ),
    ) as pool:
        for done_idx, (start, stop) in enumerate(
            pool.imap_unordered(process_index_range, chunks),
            start=1,
        ):
            if done_idx % 10 == 0 or done_idx == len(chunks):
                print(
                    f"Finished chunk {done_idx} / {len(chunks)} "
                    f"(presentations {start}:{stop})"
                )

    X_power = np.memmap(
        X_POWER_MEMMAP_PATH,
        dtype="float32",
        mode="r",
        shape=shape_x,
    )

    stim_power = np.memmap(
        STIM_POWER_MEMMAP_PATH,
        dtype="float32",
        mode="r",
        shape=shape_x,
    )

    baseline_power = np.memmap(
        BASELINE_POWER_MEMMAP_PATH,
        dtype="float32",
        mode="r",
        shape=shape_x,
    )

    return X_power, stim_power, baseline_power


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if OVERWRITE:
        remove_existing_outputs()

    print("=" * 80)
    print("Loading ViT labels")
    print("=" * 80)

    image_labels = load_vit_animate_labels(VIT_LOGITS_PATH)

    print("\n" + "=" * 80)
    print("Loading Allen session and LFP handle")
    print("=" * 80)

    cache = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)
    session = cache.get_session_data(SESSION_ID)

    probe_matches = session.probes[session.probes.description == CHOOSE_PROBE]

    if len(probe_matches) == 0:
        raise ValueError(
            f"No probe found with description={CHOOSE_PROBE}. "
            f"Available probes: {session.probes.description.tolist()}"
        )

    probe_id = int(probe_matches.index.values[0])

    print(f"Using session_id={SESSION_ID}")
    print(f"Using probe_id={probe_id}")
    print(f"Using probe description={CHOOSE_PROBE}")

    lfp = session.get_lfp(probe_id)

    time_dim, channel_dim, times, channel_ids = get_lfp_time_and_channel_info(lfp)
    fs = infer_lfp_sampling_rate(session, probe_id, lfp)

    print(f"LFP dims: {lfp.dims}")
    print(f"LFP shape: {lfp.shape}")
    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} s")
    print(f"LFP sampling rate: {fs:.6f} Hz")
    print(f"Nyquist: {fs / 2:.6f} Hz")
    print(f"Channels: {len(channel_ids)}")

    print("\n" + "=" * 80)
    print("Preparing natural-scenes table")
    print("=" * 80)

    presentations = get_natural_scenes_presentations(session)
    presentations = add_stimulus_end_times(presentations)

    image_indices_all = presentations["frame"].to_numpy(dtype=int)

    if np.any(image_indices_all < 0):
        raise ValueError("Negative image indices survived filtering")

    if image_indices_all.max() >= len(image_labels):
        raise ValueError(
            f"Max frame={image_indices_all.max()} but image_labels length={len(image_labels)}"
        )

    common_stim_duration, _ = choose_common_stimulus_duration(
        presentations=presentations,
        fs=fs,
    )

    geom = compute_padded_window_geometry(
        fs=fs,
        common_stim_duration=common_stim_duration,
    )

    presentations = filter_valid_presentations_for_padded_tensor(
        presentations=presentations,
        times=times,
        geom=geom,
    )

    image_indices = presentations["frame"].to_numpy(dtype=int)
    presentation_ids = presentations.index.values
    y = image_labels[image_indices]

    print(f"Final presentations: {len(presentations)}")
    print(f"Unique images: {len(np.unique(image_indices))}")
    print(f"Label counts [inanimate, animate]: {np.bincount(y)}")

    print("\n" + "=" * 80)
    print("Building raw disk-backed Page tensor")
    print("=" * 80)

    raw = build_raw_padded_lfp_tensor(
        lfp=lfp,
        time_dim=time_dim,
        channel_dim=channel_dim,
        times=times,
        presentations=presentations,
        geom=geom,
        n_channels=len(channel_ids),
    )

    shape_raw = raw.shape
    raw.flush()

    print("\n" + "=" * 80)
    print("Signal processing with multiprocessing")
    print("=" * 80)

    X_power, stim_power, baseline_power = run_parallel_signal_processing(
        shape_raw=shape_raw,
        fs=fs,
        geom=geom,
    )

    X_power_array = np.asarray(X_power)
    X_flat = X_power_array.reshape(X_power_array.shape[0], -1)

    print("\n" + "=" * 80)
    print("Final shapes")
    print("=" * 80)

    print(f"raw tensor:       {shape_raw}")
    print(f"X_power:          {X_power.shape}")
    print(f"X_flat:           {X_flat.shape}")
    print(f"y:                {y.shape}")
    print(f"image_indices:    {image_indices.shape}")
    print(f"presentation_ids: {presentation_ids.shape}")
    print(f"channel_ids:      {channel_ids.shape}")
    print(f"bands:            {list(BANDS.keys())}")

    print("\n" + "=" * 80)
    print("Saving final NPZ")
    print("=" * 80)

    np.savez_compressed(
        FINAL_NPZ_PATH,
        X_power=X_power_array,
        X_flat=X_flat,
        y=y,
        image_indices=image_indices,
        presentation_ids=presentation_ids,
        channel_ids=channel_ids,
        band_names=np.asarray(list(BANDS.keys())),
        stim_power=np.asarray(stim_power),
        baseline_power=np.asarray(baseline_power),
        fs_lfp=fs,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        common_stim_duration=common_stim_duration,
        padded_start_rel=geom["padded_start_rel"],
        padded_end_rel=geom["padded_end_rel"],
        padded_n_samples=geom["padded_n_samples"],
        baseline_start_idx=geom["baseline_start_idx"],
        baseline_n_samples=geom["baseline_n_samples"],
        stim_start_idx=geom["stim_start_idx"],
        stim_n_samples=geom["stim_n_samples"],
        filter_pad_sec=FILTER_PAD_SEC,
        line_freq_hz=-1.0 if LINE_FREQ_HZ is None else float(LINE_FREQ_HZ),
        highpass_hz=-1.0 if HIGHPASS_HZ is None else float(HIGHPASS_HZ),
        lowpass_hz=-1.0 if LOWPASS_HZ is None else float(LOWPASS_HZ),
        raw_memmap_path=str(RAW_MEMMAP_PATH),
        x_power_memmap_path=str(X_POWER_MEMMAP_PATH),
        stim_power_memmap_path=str(STIM_POWER_MEMMAP_PATH),
        baseline_power_memmap_path=str(BASELINE_POWER_MEMMAP_PATH),
    )

    print(f"Saved final features to: {FINAL_NPZ_PATH}")
    print("\nDone. Natural-scenes Page tensor built, filtered, and featurized.")


if __name__ == "__main__":
    # Safer when using multiprocessing.
    mp.set_start_method("spawn", force=True)
    main()