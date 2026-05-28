"""
leave_one_image_out_per_channel.py

Run leave-one-image-out classification separately for each LFP channel.

Input:
    lfp_natural_scenes_bandpower_features.npz

Expected arrays:
    X_power: shape (n_presentations, n_channels, n_bands)
    y: shape (n_presentations,)
    image_indices: shape (n_presentations,)
    channel_ids: shape (n_channels,)
    band_names: shape (n_bands,)

For each channel:
    X_ch = X_power[:, channel, :]
    Use LeaveOneGroupOut where group = image_indices
    Train logistic regression on all images except one
    Test on all presentations of the held-out image

Output:
    per_channel_leave_one_image_out_results.csv
    per_channel_leave_one_image_out_predictions.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


# =============================================================================
# Config
# =============================================================================

FEATURE_PATH = Path(
    "/media/maria/notsudata/NeuropixelDataProc/lfp_natural_scenes_tensor_probeA_session739448407/"
    "lfp_natural_scenes_bandpower_features.npz"
)

OUT_DIR = FEATURE_PATH.parent
RESULTS_CSV = OUT_DIR / "per_channel_leave_one_image_out_results.csv"
PREDICTIONS_NPZ = OUT_DIR / "per_channel_leave_one_image_out_predictions.npz"

# Logistic regression strength.
# Lower C = stronger regularization.
C = 1.0

RANDOM_STATE = 0


# =============================================================================
# Classification
# =============================================================================

def make_classifier() -> object:
    """
    Standardize bandpower features and fit L2 logistic regression.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
    )


def leave_one_group_out_predict_channel(
    X_ch: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run leave-one-group-out classification for one channel.

    Parameters
    ----------
    X_ch:
        shape (n_presentations, n_bands)

    y:
        shape (n_presentations,)

    groups:
        shape (n_presentations,)
        image index for each presentation

    Returns
    -------
    y_pred:
        shape (n_presentations,)

    y_score:
        shape (n_presentations,)
        decision-function score for class 1
    """
    logo = LeaveOneGroupOut()

    y_pred = np.full_like(y, fill_value=-1, dtype=np.int64)
    y_score = np.full(y.shape, fill_value=np.nan, dtype=np.float64)

    for train_idx, test_idx in logo.split(X_ch, y, groups=groups):
        X_train = X_ch[train_idx]
        X_test = X_ch[test_idx]

        y_train = y[train_idx]

        # Safety: logistic regression needs both classes in train.
        # With 118 images this should almost always be fine.
        if len(np.unique(y_train)) < 2:
            majority = int(np.bincount(y_train).argmax())
            y_pred[test_idx] = majority
            y_score[test_idx] = float(majority)
            continue

        clf = make_classifier()
        clf.fit(X_train, y_train)

        y_pred[test_idx] = clf.predict(X_test)

        if hasattr(clf, "decision_function"):
            y_score[test_idx] = clf.decision_function(X_test)
        else:
            y_score[test_idx] = clf.predict_proba(X_test)[:, 1]

    if np.any(y_pred < 0):
        raise RuntimeError("Some predictions were not filled")

    return y_pred, y_score


def compute_metrics(
    y: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """
    Compute pooled metrics across all held-out predictions.
    """
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)

    try:
        auc = roc_auc_score(y, y_score)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    print("=" * 80)
    print("Loading features")
    print("=" * 80)

    data = np.load(FEATURE_PATH, allow_pickle=True)

    X_power = data["X_power"]
    y = data["y"].astype(np.int64)
    groups = data["image_indices"].astype(np.int64)
    channel_ids = data["channel_ids"]
    band_names = data["band_names"]

    print(f"X_power shape:   {X_power.shape}")
    print(f"y shape:         {y.shape}")
    print(f"groups shape:    {groups.shape}")
    print(f"channel_ids:     {channel_ids.shape}")
    print(f"band_names:      {band_names}")
    print(f"Label counts:    {np.bincount(y)}")
    print(f"Unique images:   {len(np.unique(groups))}")

    n_presentations, n_channels, n_bands = X_power.shape

    all_y_pred = np.zeros((n_channels, n_presentations), dtype=np.int64)
    all_y_score = np.zeros((n_channels, n_presentations), dtype=np.float64)

    rows = []

    print("\n" + "=" * 80)
    print("Running leave-one-image-out per channel")
    print("=" * 80)

    for ch_idx in range(n_channels):
        X_ch = X_power[:, ch_idx, :]

        # Replace any remaining numerical weirdness.
        X_ch = np.nan_to_num(X_ch, nan=0.0, posinf=0.0, neginf=0.0)

        y_pred, y_score = leave_one_group_out_predict_channel(
            X_ch=X_ch,
            y=y,
            groups=groups,
        )

        metrics = compute_metrics(y, y_pred, y_score)

        channel_id = channel_ids[ch_idx]

        row = {
            "channel_index": ch_idx,
            "channel_id": channel_id,
            "n_presentations": n_presentations,
            "n_unique_images": len(np.unique(groups)),
            "n_bands": n_bands,
            **metrics,
        }

        rows.append(row)

        all_y_pred[ch_idx, :] = y_pred
        all_y_score[ch_idx, :] = y_score

        print(
            f"Channel {ch_idx:03d} | "
            f"id={channel_id} | "
            f"AUC={metrics['auc']:.3f} | "
            f"bal_acc={metrics['balanced_accuracy']:.3f} | "
            f"acc={metrics['accuracy']:.3f}"
        )

    results = pd.DataFrame(rows)
    results = results.sort_values("auc", ascending=False)

    print("\n" + "=" * 80)
    print("Top channels by AUC")
    print("=" * 80)
    print(results.head(15).to_string(index=False))

    results.to_csv(RESULTS_CSV, index=False)

    np.savez_compressed(
        PREDICTIONS_NPZ,
        y_true=y,
        image_indices=groups,
        channel_ids=channel_ids,
        band_names=band_names,
        y_pred=all_y_pred,
        y_score=all_y_score,
        results_table=results.to_records(index=False),
    )

    print("\n" + "=" * 80)
    print("Saved")
    print("=" * 80)

    print(f"Results CSV:      {RESULTS_CSV}")
    print(f"Predictions NPZ:  {PREDICTIONS_NPZ}")


if __name__ == "__main__":
    main()