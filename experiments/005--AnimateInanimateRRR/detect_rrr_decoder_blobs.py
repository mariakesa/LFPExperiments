"""
Detect coherent channel-time blobs in the RRR class-contrast decoder field.

Loads:
    rrr_W_full_weight_tensor.npy

Computes:
    W_delta[channel, time] = W[:, :, animate] - W[:, :, inanimate]

Then:
    1. Smooths W_delta using a local Gaussian filter.
    2. Thresholds positive and negative patches.
    3. Finds connected components/blobs.
    4. Saves figures and CSV summaries.

Interpretation:
    Positive blobs: channel-time regions pushing toward animate.
    Negative blobs: channel-time regions pushing toward inanimate.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, label


@dataclass
class Config:
    artifact_dir: str = "rrr_lfp_outputs"

    # Must match your RRR training script
    trial_start: float = -0.5
    trial_end: float = 0.5
    lfp_sampling_rate: float = 500.0
    downsample_factor: int = 5

    # Smoothing strength: sigma=(channels, time_bins)
    # Try (1,1), (2,2), (3,3)
    channel_sigma: float = 2.0
    time_sigma: float = 2.0

    # Thresholding
    # Example: 95 means keep top 5% absolute values after smoothing
    abs_percentile_threshold: float = 95.0

    # Remove tiny connected components
    min_blob_pixels: int = 5

    # Plotting
    output_subdir: str = "blob_figures"

    # Optional: use robust symmetric color scaling
    color_percentile: float = 99.0


cfg = Config()


def make_time_axis(n_time: int) -> np.ndarray:
    trial_window = np.arange(
        cfg.trial_start,
        cfg.trial_end,
        1.0 / cfg.lfp_sampling_rate,
    )

    if cfg.downsample_factor > 1:
        trial_window = trial_window[::cfg.downsample_factor]

    if len(trial_window) != n_time:
        print(
            "WARNING: computed time axis length does not match W time dimension.\n"
            f"Computed {len(trial_window)}, but W has n_time={n_time}.\n"
            "Using linear spacing instead."
        )
        return np.linspace(cfg.trial_start, cfg.trial_end, n_time, endpoint=False)

    return trial_window


def symmetric_vlim(A: np.ndarray, percentile: float = 99.0):
    vmax = np.percentile(np.abs(A), percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.max(np.abs(A))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def savefig(fig, fig_dir: Path, name: str):
    path = fig_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def load_W_delta(artifact_dir: Path) -> np.ndarray:
    W_path = artifact_dir / "rrr_W_full_weight_tensor.npy"

    if not W_path.exists():
        raise FileNotFoundError(
            f"Could not find {W_path}\n"
            "Expected saved tensor with shape channels x time x class."
        )

    W = np.load(W_path)

    if W.ndim != 3:
        raise ValueError(f"Expected W with 3 dims: channels x time x class, got {W.shape}")

    if W.shape[2] != 2:
        raise ValueError(
            f"Expected binary classifier with 2 classes, got W.shape={W.shape}"
        )

    W_delta = W[:, :, 1] - W[:, :, 0]

    print(f"Loaded W: {W.shape} = channels x time x class")
    print(f"Computed W_delta: {W_delta.shape} = channels x time")
    print("Convention: W_delta > 0 pushes animate; W_delta < 0 pushes inanimate")

    return W_delta


def smooth_decoder_field(W_delta: np.ndarray) -> np.ndarray:
    W_smooth = gaussian_filter(
        W_delta,
        sigma=(cfg.channel_sigma, cfg.time_sigma),
        mode="nearest",
    )

    print(
        f"Applied Gaussian smoothing with "
        f"sigma=(channel={cfg.channel_sigma}, time={cfg.time_sigma})"
    )

    return W_smooth


def make_blob_masks(W_smooth: np.ndarray):
    threshold = np.percentile(np.abs(W_smooth), cfg.abs_percentile_threshold)

    pos_mask = W_smooth >= threshold
    neg_mask = W_smooth <= -threshold

    print(f"Absolute threshold percentile: {cfg.abs_percentile_threshold}")
    print(f"Threshold value: {threshold:.6f}")
    print(f"Positive mask pixels: {pos_mask.sum()}")
    print(f"Negative mask pixels: {neg_mask.sum()}")

    return pos_mask, neg_mask, threshold


def connected_components(mask: np.ndarray):
    """
    8-connected components in 2D.

    Structure:
        [[1,1,1],
         [1,1,1],
         [1,1,1]]
    means diagonal neighbors count as connected.
    """
    structure = np.ones((3, 3), dtype=int)
    labeled, n_components = label(mask, structure=structure)
    return labeled, n_components


def summarize_blobs(
    labeled: np.ndarray,
    n_components: int,
    W_smooth: np.ndarray,
    W_raw: np.ndarray,
    time_axis: np.ndarray,
    sign: str,
) -> pd.DataFrame:
    rows = []

    for blob_id in range(1, n_components + 1):
        coords = np.argwhere(labeled == blob_id)

        if coords.shape[0] < cfg.min_blob_pixels:
            continue

        channels = coords[:, 0]
        time_idx = coords[:, 1]

        smooth_values = W_smooth[channels, time_idx]
        raw_values = W_raw[channels, time_idx]

        weights = np.abs(smooth_values)
        if weights.sum() <= 0:
            center_channel = float(np.mean(channels))
            center_time = float(np.mean(time_axis[time_idx]))
        else:
            center_channel = float(np.average(channels, weights=weights))
            center_time = float(np.average(time_axis[time_idx], weights=weights))

        peak_local_idx = np.argmax(np.abs(smooth_values))
        peak_channel = int(channels[peak_local_idx])
        peak_time = float(time_axis[time_idx[peak_local_idx]])
        peak_value = float(smooth_values[peak_local_idx])

        row = {
            "sign": sign,
            "blob_id": blob_id,
            "n_pixels": int(coords.shape[0]),
            "center_channel": center_channel,
            "center_time": center_time,
            "peak_channel": peak_channel,
            "peak_time": peak_time,
            "peak_smooth_value": peak_value,
            "smooth_signed_mass": float(smooth_values.sum()),
            "smooth_abs_mass": float(np.abs(smooth_values).sum()),
            "raw_signed_mass": float(raw_values.sum()),
            "raw_abs_mass": float(np.abs(raw_values).sum()),
            "channel_min": int(channels.min()),
            "channel_max": int(channels.max()),
            "time_min": float(time_axis[time_idx].min()),
            "time_max": float(time_axis[time_idx].max()),
            "mean_smooth_value": float(smooth_values.mean()),
            "mean_raw_value": float(raw_values.mean()),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values("smooth_abs_mass", ascending=False).reset_index(drop=True)

    return df


def plot_raw_and_smoothed(W_delta: np.ndarray, W_smooth: np.ndarray, time_axis: np.ndarray, fig_dir: Path):
    n_channels, _ = W_delta.shape

    vmin_raw, vmax_raw = symmetric_vlim(W_delta, cfg.color_percentile)
    vmin_smooth, vmax_smooth = symmetric_vlim(W_smooth, cfg.color_percentile)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        W_delta,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
        vmin=vmin_raw,
        vmax=vmax_raw,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Channel index")
    ax.set_title("Raw RRR class-contrast decoder field")
    fig.colorbar(im, ax=ax, label="W_animate - W_inanimate")
    savefig(fig, fig_dir, "01_raw_W_delta.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        W_smooth,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
        vmin=vmin_smooth,
        vmax=vmax_smooth,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Channel index")
    ax.set_title(
        f"Smoothed class-contrast decoder field "
        f"σ=({cfg.channel_sigma}, {cfg.time_sigma})"
    )
    fig.colorbar(im, ax=ax, label="Smoothed W_animate - W_inanimate")
    savefig(fig, fig_dir, "02_smoothed_W_delta.png")


def plot_blob_masks(
    W_smooth: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    threshold: float,
    time_axis: np.ndarray,
    fig_dir: Path,
):
    n_channels, _ = W_smooth.shape

    blob_overlay = np.zeros_like(W_smooth, dtype=float)
    blob_overlay[pos_mask] = 1.0
    blob_overlay[neg_mask] = -1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        blob_overlay,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
        vmin=-1,
        vmax=1,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Channel index")
    ax.set_title(
        f"Thresholded positive/negative blob mask "
        f"|W_smooth| >= {cfg.abs_percentile_threshold}th percentile"
    )
    fig.colorbar(im, ax=ax, label="-1 negative blob, +1 positive blob")
    savefig(fig, fig_dir, "03_thresholded_blob_mask.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = symmetric_vlim(W_smooth, cfg.color_percentile)
    im = ax.imshow(
        W_smooth,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
        vmin=vmin,
        vmax=vmax,
    )

    # Overlay contours
    ax.contour(
        time_axis,
        np.arange(n_channels),
        pos_mask.astype(float),
        levels=[0.5],
        linewidths=1.0,
    )
    ax.contour(
        time_axis,
        np.arange(n_channels),
        neg_mask.astype(float),
        levels=[0.5],
        linewidths=1.0,
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Channel index")
    ax.set_title("Smoothed decoder field with detected blob contours")
    fig.colorbar(im, ax=ax, label="Smoothed W_animate - W_inanimate")
    savefig(fig, fig_dir, "04_smoothed_W_delta_with_blob_contours.png")


def plot_labeled_components(
    labeled: np.ndarray,
    n_components: int,
    time_axis: np.ndarray,
    fig_dir: Path,
    sign: str,
):
    n_channels, _ = labeled.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        labeled,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Channel index")
    ax.set_title(f"Labeled connected components: {sign} blobs, n={n_components}")
    fig.colorbar(im, ax=ax, label="Component id")
    savefig(fig, fig_dir, f"05_labeled_{sign}_components.png")


def plot_blob_centers(
    W_smooth: np.ndarray,
    blob_df: pd.DataFrame,
    time_axis: np.ndarray,
    fig_dir: Path,
):
    if blob_df.empty:
        print("No blobs to plot centers for.")
        return

    n_channels, _ = W_smooth.shape
    vmin, vmax = symmetric_vlim(W_smooth, cfg.color_percentile)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        W_smooth,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
        vmin=vmin,
        vmax=vmax,
    )

    for _, row in blob_df.iterrows():
        marker = "o" if row["sign"] == "positive" else "x"
        ax.scatter(
            row["center_time"],
            row["center_channel"],
            s=80,
            marker=marker,
            linewidths=1.5,
        )
        ax.text(
            row["center_time"],
            row["center_channel"],
            f"{row['sign'][0].upper()}{int(row['blob_id'])}",
            fontsize=8,
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Channel index")
    ax.set_title("Detected blob centers on smoothed decoder field")
    fig.colorbar(im, ax=ax, label="Smoothed W_animate - W_inanimate")
    savefig(fig, fig_dir, "06_blob_centers_on_smoothed_W_delta.png")


def main():
    artifact_dir = Path(cfg.artifact_dir)
    fig_dir = artifact_dir / cfg.output_subdir
    fig_dir.mkdir(exist_ok=True, parents=True)

    W_delta = load_W_delta(artifact_dir)
    n_channels, n_time = W_delta.shape
    time_axis = make_time_axis(n_time)

    W_smooth = smooth_decoder_field(W_delta)

    pos_mask, neg_mask, threshold = make_blob_masks(W_smooth)

    pos_labeled, n_pos = connected_components(pos_mask)
    neg_labeled, n_neg = connected_components(neg_mask)

    print(f"Connected positive components before size filtering: {n_pos}")
    print(f"Connected negative components before size filtering: {n_neg}")

    pos_df = summarize_blobs(
        labeled=pos_labeled,
        n_components=n_pos,
        W_smooth=W_smooth,
        W_raw=W_delta,
        time_axis=time_axis,
        sign="positive",
    )

    neg_df = summarize_blobs(
        labeled=neg_labeled,
        n_components=n_neg,
        W_smooth=W_smooth,
        W_raw=W_delta,
        time_axis=time_axis,
        sign="negative",
    )

    blob_df = pd.concat([pos_df, neg_df], ignore_index=True)

    if not blob_df.empty:
        blob_df = blob_df.sort_values("smooth_abs_mass", ascending=False).reset_index(drop=True)

    summary_path = artifact_dir / "rrr_smoothed_decoder_blob_summary.csv"
    blob_df.to_csv(summary_path, index=False)

    print("\nBlob summary:")
    if blob_df.empty:
        print("No blobs survived filtering.")
    else:
        print(blob_df.head(20))

    print(f"\nSaved blob summary to {summary_path}")

    # Save arrays too
    np.save(artifact_dir / "rrr_W_delta_raw.npy", W_delta)
    np.save(artifact_dir / "rrr_W_delta_smooth.npy", W_smooth)
    np.save(artifact_dir / "rrr_positive_blob_mask.npy", pos_mask)
    np.save(artifact_dir / "rrr_negative_blob_mask.npy", neg_mask)
    np.save(artifact_dir / "rrr_positive_labeled_components.npy", pos_labeled)
    np.save(artifact_dir / "rrr_negative_labeled_components.npy", neg_labeled)

    # Figures
    plot_raw_and_smoothed(W_delta, W_smooth, time_axis, fig_dir)
    plot_blob_masks(W_smooth, pos_mask, neg_mask, threshold, time_axis, fig_dir)
    plot_labeled_components(pos_labeled, n_pos, time_axis, fig_dir, sign="positive")
    plot_labeled_components(neg_labeled, n_neg, time_axis, fig_dir, sign="negative")
    plot_blob_centers(W_smooth, blob_df, time_axis, fig_dir)

    print("\nSaved arrays:")
    print(f"  {artifact_dir / 'rrr_W_delta_raw.npy'}")
    print(f"  {artifact_dir / 'rrr_W_delta_smooth.npy'}")
    print(f"  {artifact_dir / 'rrr_positive_blob_mask.npy'}")
    print(f"  {artifact_dir / 'rrr_negative_blob_mask.npy'}")
    print(f"  {artifact_dir / 'rrr_positive_labeled_components.npy'}")
    print(f"  {artifact_dir / 'rrr_negative_labeled_components.npy'}")

    print("\nSaved figures to:")
    print(fig_dir.resolve())

    print("\nDone. Blob goblin detector finished.")


if __name__ == "__main__":
    main()