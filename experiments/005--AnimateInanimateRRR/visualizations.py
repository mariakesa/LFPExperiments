"""
Visualize low-rank RRR artifacts from the LFP animate/inanimate decoder.

This script visualizes the learned decomposition:

    W[channel, time, class] = sum_r U[channel, r] * V[r, time, class]

For binary animate/inanimate decoding, the most interpretable object is the
class contrast:

    W_delta[channel, time] = W[:, :, animate] - W[:, :, inanimate]

Positive W_delta means:
    larger normalized LFP values at that channel/time push the model toward animate.

Negative W_delta means:
    larger normalized LFP values at that channel/time push the model toward inanimate.

Important:
    Individual U and V signs are not uniquely identifiable.
    The product U[:, r] * V[r, :, :] is meaningful; U alone or V alone can flip sign.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Config:
    artifact_dir: str = "rrr_lfp_outputs"

    # Must match the training script.
    trial_start: float = -0.5
    trial_end: float = 0.5
    lfp_sampling_rate: float = 500.0
    downsample_factor: int = 5

    # Plot settings
    max_rank_plots: int = 10
    output_subdir: str = "figures"


cfg = Config()


def load_artifacts(artifact_dir: Path):
    U_path = artifact_dir / "rrr_U_channel_basis.npy"
    V_path = artifact_dir / "rrr_V_temporal_basis.npy"
    W_path = artifact_dir / "rrr_W_full_weight_tensor.npy"

    if not U_path.exists():
        raise FileNotFoundError(f"Missing {U_path}")
    if not V_path.exists():
        raise FileNotFoundError(f"Missing {V_path}")

    U = np.load(U_path)
    V = np.load(V_path)

    if W_path.exists():
        W = np.load(W_path)
    else:
        print("Full W tensor not found. Reconstructing W from U and V.")
        W = np.einsum("nr,rtd->ntd", U, V)

    print("Loaded artifacts:")
    print(f"  U shape: {U.shape}  = channels x rank")
    print(f"  V shape: {V.shape}  = rank x time x class")
    print(f"  W shape: {W.shape}  = channels x time x class")

    return U, V, W


def make_time_axis(n_time: int):
    full_trial_window = np.arange(
        cfg.trial_start,
        cfg.trial_end,
        1.0 / cfg.lfp_sampling_rate,
    )

    if cfg.downsample_factor > 1:
        full_trial_window = full_trial_window[::cfg.downsample_factor]

    if len(full_trial_window) != n_time:
        print(
            "WARNING: computed time axis length does not match V/W time dimension.\n"
            f"Computed {len(full_trial_window)}, but model has {n_time}.\n"
            "Using linear spacing instead."
        )
        return np.linspace(cfg.trial_start, cfg.trial_end, n_time, endpoint=False)

    return full_trial_window


def savefig(fig, fig_dir: Path, name: str):
    fig_path = fig_dir / name
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


def plot_training_curves(artifact_dir: Path, fig_dir: Path):
    path = artifact_dir / "rrr_lfp_training_history.csv"
    if not path.exists():
        print(f"Skipping training curves; missing {path}")
        return

    hist = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["epoch"], hist["train_loss"], label="Train loss")
    ax.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("RRR training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, fig_dir, "01_training_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["epoch"], hist["train_accuracy"], label="Train accuracy")
    ax.plot(hist["epoch"], hist["val_accuracy"], label="Validation accuracy")
    if "val_auc" in hist.columns:
        ax.plot(hist["epoch"], hist["val_auc"], label="Validation AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.05)
    ax.set_title("RRR classification metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, fig_dir, "02_training_metrics.png")


def plot_validation_predictions(artifact_dir: Path, fig_dir: Path):
    path = artifact_dir / "rrr_lfp_validation_predictions.csv"
    if not path.exists():
        print(f"Skipping validation prediction plots; missing {path}")
        return

    pred = pd.read_csv(path)

    if "p_animate" not in pred.columns:
        print("Skipping prediction plot; p_animate column not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    for label_value, label_name in [(0, "Inanimate"), (1, "Animate")]:
        vals = pred.loc[pred["y_true"] == label_value, "p_animate"]
        ax.hist(vals, bins=bins, alpha=0.6, label=label_name)

    ax.set_xlabel("Predicted P(animate)")
    ax.set_ylabel("Validation trial count")
    ax.set_title("Validation predictions by true class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, fig_dir, "03_validation_probability_histogram.png")


def plot_U_channel_basis(U: np.ndarray, fig_dir: Path):
    n_channels, rank = U.shape

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(U.T, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Rank component")
    ax.set_title("U channel basis: rank components across LFP channels")
    fig.colorbar(im, ax=ax, label="U weight")
    savefig(fig, fig_dir, "04_U_channel_basis_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    channel_importance = np.linalg.norm(U, axis=1)
    ax.plot(np.arange(n_channels), channel_importance)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("||U[channel, :]||")
    ax.set_title("Channel importance from low-rank basis")
    ax.grid(True, alpha=0.3)
    savefig(fig, fig_dir, "05_channel_importance_from_U.png")


def plot_V_temporal_basis(V: np.ndarray, time_axis: np.ndarray, fig_dir: Path):
    rank, n_time, n_classes = V.shape

    if n_classes != 2:
        print(f"Expected binary classifier with 2 classes, got {n_classes}. Still plotting all classes.")

    max_r = min(rank, cfg.max_rank_plots)

    for r in range(max_r):
        fig, ax = plt.subplots(figsize=(8, 4))

        for c in range(n_classes):
            label = "Inanimate" if c == 0 else "Animate" if c == 1 else f"Class {c}"
            ax.plot(time_axis, V[r, :, c], label=label)

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Time from image onset (s)")
        ax.set_ylabel("V weight")
        ax.set_title(f"Temporal basis V for rank {r}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, fig_dir, f"06_V_temporal_basis_rank_{r:02d}.png")

    if n_classes == 2:
        delta_V = V[:, :, 1] - V[:, :, 0]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            delta_V,
            aspect="auto",
            interpolation="nearest",
            extent=[time_axis[0], time_axis[-1], rank - 0.5, -0.5],
        )
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Time from image onset (s)")
        ax.set_ylabel("Rank component")
        ax.set_title("Class-contrast temporal basis: V_animate - V_inanimate")
        fig.colorbar(im, ax=ax, label="Delta V")
        savefig(fig, fig_dir, "07_delta_V_temporal_basis_heatmap.png")


def symmetric_vlim(A: np.ndarray, percentile: float = 99.0):
    vmax = np.percentile(np.abs(A), percentile)
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = np.max(np.abs(A))
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    return -vmax, vmax


def plot_rank1_discriminative_patterns(U: np.ndarray, V: np.ndarray, time_axis: np.ndarray, fig_dir: Path):
    """
    Plot each rank-1 contribution to the animate-vs-inanimate decoder:

        W_delta_r[channel, time] = U[channel, r] * (V[r, time, animate] - V[r, time, inanimate])
    """
    n_channels, rank = U.shape
    _, n_time, n_classes = V.shape

    if n_classes != 2:
        print("Skipping rank-1 discriminative patterns; expected 2 output classes.")
        return

    delta_V = V[:, :, 1] - V[:, :, 0]

    max_r = min(rank, cfg.max_rank_plots)

    for r in range(max_r):
        component = np.outer(U[:, r], delta_V[r, :])  # channels x time
        vmin, vmax = symmetric_vlim(component)

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(
            component,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Time from image onset (s)")
        ax.set_ylabel("Channel index")
        ax.set_title(f"Rank-{r} class-contrast pattern: U[:, {r}] outer ΔV[{r}, :]")
        fig.colorbar(im, ax=ax, label="Contribution to animate logit minus inanimate logit")
        savefig(fig, fig_dir, f"08_rank1_discriminative_pattern_{r:02d}.png")


def plot_full_class_contrast(W: np.ndarray, time_axis: np.ndarray, fig_dir: Path):
    """
    Plot the full animate-vs-inanimate decoder sensitivity:

        W_delta = W[:, :, animate] - W[:, :, inanimate]
    """
    n_channels, n_time, n_classes = W.shape

    if n_classes != 2:
        print("Skipping full class contrast; expected 2 output classes.")
        return

    W_delta = W[:, :, 1] - W[:, :, 0]

    vmin, vmax = symmetric_vlim(W_delta)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        W_delta,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], 0, n_channels - 1],
        vmin=vmin,
        vmax=vmax,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Channel index")
    ax.set_title("Full low-rank class-contrast decoder: W_animate - W_inanimate")
    fig.colorbar(im, ax=ax, label="Animate sensitivity minus inanimate sensitivity")
    savefig(fig, fig_dir, "09_full_W_class_contrast_heatmap.png")

    temporal_importance = np.linalg.norm(W_delta, axis=0)
    channel_importance = np.linalg.norm(W_delta, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, temporal_importance)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("||W_delta[:, time]||")
    ax.set_title("When the decoder is sensitive to animate/inanimate information")
    ax.grid(True, alpha=0.3)
    savefig(fig, fig_dir, "10_temporal_importance_from_W_delta.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(n_channels), channel_importance)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("||W_delta[channel, :]||")
    ax.set_title("Which channels contribute to animate/inanimate decoding")
    ax.grid(True, alpha=0.3)
    savefig(fig, fig_dir, "11_channel_importance_from_W_delta.png")

    # Save numeric summaries too.
    return W_delta, temporal_importance, channel_importance


def save_summary_tables(
    artifact_dir: Path,
    U: np.ndarray,
    V: np.ndarray,
    W_delta: np.ndarray,
    temporal_importance: np.ndarray,
    channel_importance: np.ndarray,
    time_axis: np.ndarray,
):
    rank = U.shape[1]

    rank_rows = []
    if V.shape[2] == 2:
        delta_V = V[:, :, 1] - V[:, :, 0]

        for r in range(rank):
            rank_component = np.outer(U[:, r], delta_V[r, :])
            rank_rows.append({
                "rank": r,
                "U_norm": float(np.linalg.norm(U[:, r])),
                "delta_V_norm": float(np.linalg.norm(delta_V[r, :])),
                "rank_component_norm": float(np.linalg.norm(rank_component)),
                "peak_abs_time": float(time_axis[np.argmax(np.abs(delta_V[r, :]))]),
                "peak_abs_delta_V": float(delta_V[r, np.argmax(np.abs(delta_V[r, :]))]),
            })

    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(artifact_dir / "rrr_rank_component_summary.csv", index=False)

    temporal_df = pd.DataFrame({
        "time_from_presentation_onset": time_axis,
        "temporal_importance": temporal_importance,
    })
    temporal_df.to_csv(artifact_dir / "rrr_temporal_importance.csv", index=False)

    channel_df = pd.DataFrame({
        "channel_index": np.arange(len(channel_importance)),
        "channel_importance": channel_importance,
        "U_norm": np.linalg.norm(U, axis=1),
    })
    channel_df.to_csv(artifact_dir / "rrr_channel_importance.csv", index=False)

    print(f"Saved {artifact_dir / 'rrr_rank_component_summary.csv'}")
    print(f"Saved {artifact_dir / 'rrr_temporal_importance.csv'}")
    print(f"Saved {artifact_dir / 'rrr_channel_importance.csv'}")


def main():
    artifact_dir = Path(cfg.artifact_dir)
    fig_dir = artifact_dir / cfg.output_subdir
    fig_dir.mkdir(exist_ok=True, parents=True)

    U, V, W = load_artifacts(artifact_dir)

    n_channels, rank = U.shape
    rank_v, n_time, n_classes = V.shape

    if rank_v != rank:
        raise ValueError(f"U rank {rank} and V rank {rank_v} do not match.")

    time_axis = make_time_axis(n_time)

    plot_training_curves(artifact_dir, fig_dir)
    plot_validation_predictions(artifact_dir, fig_dir)

    plot_U_channel_basis(U, fig_dir)
    plot_V_temporal_basis(V, time_axis, fig_dir)
    plot_rank1_discriminative_patterns(U, V, time_axis, fig_dir)

    result = plot_full_class_contrast(W, time_axis, fig_dir)

    if result is not None:
        W_delta, temporal_importance, channel_importance = result
        save_summary_tables(
            artifact_dir=artifact_dir,
            U=U,
            V=V,
            W_delta=W_delta,
            temporal_importance=temporal_importance,
            channel_importance=channel_importance,
            time_axis=time_axis,
        )

    print("\nDone. Figures saved to:")
    print(fig_dir.resolve())

def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation with protection against constant vectors.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def plot_V_temporal_basis(V: np.ndarray, time_axis: np.ndarray, fig_dir: Path):
    """
    Plot class-specific temporal bases V for each rank.

    For binary classification:
        V[r, :, 0] = inanimate temporal basis
        V[r, :, 1] = animate temporal basis

    Also computes:
        corr(V_inanimate, V_animate)

    and saves:
        - one overlay plot per rank
        - one summary bar plot of correlations
        - one CSV table of correlations
        - heatmaps for V_inanimate, V_animate, and delta V
    """
    rank, n_time, n_classes = V.shape

    max_r = min(rank, cfg.max_rank_plots)

    corr_rows = []

    # -----------------------
    # Individual rank plots
    # -----------------------
    for r in range(max_r):
        fig, ax = plt.subplots(figsize=(8, 4))

        if n_classes == 2:
            v_inanimate = V[r, :, 0]
            v_animate = V[r, :, 1]
            rho = safe_corrcoef(v_inanimate, v_animate)

            corr_rows.append({
                "rank": r,
                "corr_inanimate_animate": rho,
                "inanimate_norm": float(np.linalg.norm(v_inanimate)),
                "animate_norm": float(np.linalg.norm(v_animate)),
                "delta_norm": float(np.linalg.norm(v_animate - v_inanimate)),
            })

            ax.plot(time_axis, v_inanimate, label="Inanimate V")
            ax.plot(time_axis, v_animate, label="Animate V")
            ax.plot(
                time_axis,
                v_animate - v_inanimate,
                linestyle="--",
                label="Animate - inanimate",
            )

            title_rho = "nan" if np.isnan(rho) else f"{rho:.3f}"
            ax.set_title(
                f"Temporal basis V for rank {r} | corr(inanimate, animate) = {title_rho}"
            )

        else:
            for c in range(n_classes):
                ax.plot(time_axis, V[r, :, c], label=f"Class {c}")

            ax.set_title(f"Temporal basis V for rank {r}")

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Time from image onset (s)")
        ax.set_ylabel("V weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

        savefig(fig, fig_dir, f"06_V_temporal_basis_rank_{r:02d}_with_corr.png")

    # -----------------------
    # Binary-only summaries
    # -----------------------
    if n_classes != 2:
        print(f"Skipping binary V correlation summaries because n_classes={n_classes}")
        return

    corr_df = pd.DataFrame(corr_rows)
    corr_path = fig_dir.parent / "rrr_V_animate_inanimate_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved {corr_path}")

    # -----------------------
    # Correlation bar plot
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(corr_df["rank"], corr_df["corr_inanimate_animate"])
    ax.axhline(0, linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Rank component")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Correlation between inanimate and animate temporal bases")
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, fig_dir, "07_V_animate_inanimate_correlation_by_rank.png")

    # -----------------------
    # Heatmaps: inanimate, animate, delta
    # -----------------------
    V_inanimate = V[:, :, 0]
    V_animate = V[:, :, 1]
    delta_V = V_animate - V_inanimate

    vmin, vmax = symmetric_vlim(V)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        V_inanimate,
        aspect="auto",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], rank - 0.5, -0.5],
        vmin=vmin,
        vmax=vmax,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Rank component")
    ax.set_title("Inanimate temporal basis V[:, :, 0]")
    fig.colorbar(im, ax=ax, label="V weight")
    savefig(fig, fig_dir, "08_V_inanimate_temporal_basis_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        V_animate,
        aspect="auto",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], rank - 0.5, -0.5],
        vmin=vmin,
        vmax=vmax,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Rank component")
    ax.set_title("Animate temporal basis V[:, :, 1]")
    fig.colorbar(im, ax=ax, label="V weight")
    savefig(fig, fig_dir, "09_V_animate_temporal_basis_heatmap.png")

    vmin_delta, vmax_delta = symmetric_vlim(delta_V)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        delta_V,
        aspect="auto",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], rank - 0.5, -0.5],
        vmin=vmin_delta,
        vmax=vmax_delta,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time from image onset (s)")
    ax.set_ylabel("Rank component")
    ax.set_title("Class-contrast temporal basis: V_animate - V_inanimate")
    fig.colorbar(im, ax=ax, label="Delta V")
    savefig(fig, fig_dir, "10_delta_V_temporal_basis_heatmap.png")


if __name__ == "__main__":
    main()