"""
Permutation test for per-channel LFP animate/inanimate decoding.

This script assumes you already created:

    lfp_natural_scenes_bandpower_features.npz

with keys:
    X_power:       shape (n_presentations, n_channels, n_bands)
    y:             shape (n_presentations,)
    image_indices: shape (n_presentations,)
    channel_ids
    band_names

Important:
    The permutation is done at the IMAGE level, not the presentation level.

Why:
    There are repeated presentations of the same natural scene image.
    To avoid destroying the repeated-image structure, each shuffled image label
    is propagated to all presentations of that image.

Test:
    For each channel:
        X_channel = X_power[:, channel, :]
        decoder = StandardScaler + LogisticRegression
        CV = leave-one-image-out, implemented as LeaveOneGroupOut
        metric = ROC AUC

P-value:
    p = (1 + number of permuted AUCs >= true AUC) / (1 + n_permutations)

Outputs:
    per_channel_permutation_results.csv
    per_channel_permutation_nulls.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Config
# =============================================================================

FEATURE_PATH = (
    "/media/maria/notsudata/NeuropixelDataProc/"
    "lfp_natural_scenes_tensor_probeA_session739448407/"
    "lfp_natural_scenes_bandpower_features.npz"
)

OUT_DIR = Path(
    "/media/maria/notsudata/NeuropixelDataProc/"
    "lfp_natural_scenes_tensor_probeA_session739448407/"
)

RESULTS_CSV = OUT_DIR / "per_channel_image_level_permutation_results.csv"
NULLS_NPZ = OUT_DIR / "per_channel_image_level_permutation_nulls.npz"

N_PERMUTATIONS = 1000
RANDOM_SEED = 123

# Logistic regression settings.
C = 1.0
MAX_ITER = 5000


# =============================================================================
# Helpers
# =============================================================================

def make_decoder() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=MAX_ITER,
        ),
    )


def leave_one_image_out_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """
    Return cross-validated predicted probabilities for class 1.

    Uses LeaveOneGroupOut where group = image identity.
    """
    cv = LeaveOneGroupOut()
    clf = make_decoder()

    prob = cross_val_predict(
        clf,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    return prob


def compute_metrics_from_prob(
    y_true: np.ndarray,
    prob: np.ndarray,
) -> dict:
    """
    Compute AUC, balanced accuracy, and accuracy.

    Class prediction threshold is 0.5.
    """
    y_pred = (prob >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true, prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def make_image_level_permuted_y(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shuffle labels at the image level, then map back to presentations.

    This preserves repeated-image structure.

    Assumes each image has one true label across all its presentations.
    """
    unique_images = np.unique(groups)

    image_to_label = {}

    for img in unique_images:
        labels_for_img = np.unique(y[groups == img])

        if len(labels_for_img) != 1:
            raise ValueError(
                f"Image {img} has multiple labels: {labels_for_img}. "
                "Check your label construction."
            )

        image_to_label[img] = int(labels_for_img[0])

    true_image_labels = np.array(
        [image_to_label[img] for img in unique_images],
        dtype=int,
    )

    shuffled_image_labels = rng.permutation(true_image_labels)

    shuffled_lookup = {
        img: label
        for img, label in zip(unique_images, shuffled_image_labels)
    }

    y_perm = np.array(
        [shuffled_lookup[img] for img in groups],
        dtype=int,
    )

    return y_perm


def summarize_null(true_auc: float, null_auc: np.ndarray) -> dict:
    """
    One-sided empirical p-value for AUC being larger than chance/null.
    """
    p_value = (1.0 + np.sum(null_auc >= true_auc)) / (1.0 + len(null_auc))

    return {
        "p_value": p_value,
        "null_mean_auc": float(np.mean(null_auc)),
        "null_std_auc": float(np.std(null_auc, ddof=1)),
        "null_p95_auc": float(np.percentile(null_auc, 95)),
        "null_p99_auc": float(np.percentile(null_auc, 99)),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("Loading features")
    print("=" * 80)

    data = np.load(FEATURE_PATH, allow_pickle=True)

    X_power = data["X_power"]
    y = data["y"].astype(int)
    groups = data["image_indices"].astype(int)
    channel_ids = data["channel_ids"]
    band_names = data["band_names"]

    n_presentations, n_channels, n_bands = X_power.shape
    unique_images = np.unique(groups)

    print(f"X_power shape:   {X_power.shape}")
    print(f"y shape:         {y.shape}")
    print(f"groups shape:    {groups.shape}")
    print(f"channel_ids:     {channel_ids.shape}")
    print(f"band_names:      {band_names}")
    print(f"Label counts:    {np.bincount(y)}")
    print(f"Unique images:   {len(unique_images)}")
    print(f"Permutations:    {N_PERMUTATIONS}")

    # Sanity check: every image should have exactly one label.
    for img in unique_images:
        labels = np.unique(y[groups == img])
        if len(labels) != 1:
            raise ValueError(f"Image {img} has multiple labels: {labels}")

    rng = np.random.default_rng(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("Computing true per-channel scores")
    print("=" * 80)

    true_auc = np.zeros(n_channels, dtype=float)
    true_bal_acc = np.zeros(n_channels, dtype=float)
    true_acc = np.zeros(n_channels, dtype=float)

    # Store true CV probabilities if you want to inspect predictions later.
    true_probs = np.zeros((n_presentations, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        X_ch = X_power[:, ch, :]

        prob = leave_one_image_out_predictions(
            X=X_ch,
            y=y,
            groups=groups,
        )

        true_probs[:, ch] = prob.astype(np.float32)

        metrics = compute_metrics_from_prob(y, prob)

        true_auc[ch] = metrics["auc"]
        true_bal_acc[ch] = metrics["balanced_accuracy"]
        true_acc[ch] = metrics["accuracy"]

        print(
            f"Channel {ch:03d} | id={channel_ids[ch]} | "
            f"AUC={true_auc[ch]:.4f} | "
            f"bal_acc={true_bal_acc[ch]:.4f} | "
            f"acc={true_acc[ch]:.4f}"
        )

    print("\n" + "=" * 80)
    print("Running image-level permutation test")
    print("=" * 80)

    null_auc = np.zeros((N_PERMUTATIONS, n_channels), dtype=np.float32)

    for perm_idx in range(N_PERMUTATIONS):
        y_perm = make_image_level_permuted_y(
            y=y,
            groups=groups,
            rng=rng,
        )

        for ch in range(n_channels):
            X_ch = X_power[:, ch, :]

            prob_perm = leave_one_image_out_predictions(
                X=X_ch,
                y=y_perm,
                groups=groups,
            )

            null_auc[perm_idx, ch] = roc_auc_score(y_perm, prob_perm)

        if (perm_idx + 1) % 10 == 0 or (perm_idx + 1) == N_PERMUTATIONS:
            best_p_so_far = np.min(
                [
                    (1.0 + np.sum(null_auc[: perm_idx + 1, ch] >= true_auc[ch]))
                    / (1.0 + perm_idx + 1)
                    for ch in range(n_channels)
                ]
            )

            print(
                f"Finished permutation {perm_idx + 1} / {N_PERMUTATIONS} | "
                f"best p so far={best_p_so_far:.4g}"
            )

    print("\n" + "=" * 80)
    print("Summarizing p-values")
    print("=" * 80)

    rows = []

    p_values = np.zeros(n_channels, dtype=float)

    for ch in range(n_channels):
        null_ch = null_auc[:, ch].astype(float)

        summary = summarize_null(
            true_auc=true_auc[ch],
            null_auc=null_ch,
        )

        p_values[ch] = summary["p_value"]

        rows.append(
            {
                "channel_index": ch,
                "channel_id": channel_ids[ch],
                "n_presentations": n_presentations,
                "n_unique_images": len(unique_images),
                "n_bands": n_bands,
                "true_auc": true_auc[ch],
                "true_balanced_accuracy": true_bal_acc[ch],
                "true_accuracy": true_acc[ch],
                **summary,
            }
        )

    reject, q_values, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="fdr_bh",
    )

    results = pd.DataFrame(rows)
    results["q_value_fdr_bh"] = q_values
    results["reject_fdr_0.05"] = reject

    results = results.sort_values(
        ["p_value", "true_auc"],
        ascending=[True, False],
    )

    print(results.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("Saving")
    print("=" * 80)

    results.to_csv(RESULTS_CSV, index=False)

    np.savez_compressed(
        NULLS_NPZ,
        null_auc=null_auc,
        true_auc=true_auc,
        true_balanced_accuracy=true_bal_acc,
        true_accuracy=true_acc,
        true_probs=true_probs,
        y=y,
        groups=groups,
        channel_ids=channel_ids,
        band_names=band_names,
        n_permutations=N_PERMUTATIONS,
        random_seed=RANDOM_SEED,
    )

    print(f"Results CSV: {RESULTS_CSV}")
    print(f"Nulls NPZ:   {NULLS_NPZ}")


if __name__ == "__main__":
    main()