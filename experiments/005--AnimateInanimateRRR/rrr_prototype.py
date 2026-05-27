"""
Fit a single-session reduced-rank regression / logistic decoder on Allen Neuropixels LFP trials.

This version derives animate/inanimate labels from ViT logits:

    top1 = argmax(ViT logits)
    y_image = 1 if top1 <= 397 else 0

Assumption:
    presentation_table["frame"] indexes the natural-scenes image used in the ViT logits array.

Model:
    X has shape: trials x time x channels
    y has shape: trials

RRR decoder:
    W[channel, time, class] = sum_r U[channel, r] * V[r, time, class]
"""

from dataclasses import dataclass
from pathlib import Path
import os
import random

from dotenv import load_dotenv

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


# -----------------------
# Config
# -----------------------

@dataclass
class Config:
    # Allen cache
    cache_path_env: str = "LFP_DATA_PATH"

    # Session/probe
    #session_id: int = 831882777
    #session_id: int = 715093703
    session_id: int = 739448407
    choose_probe: str = "probeA"
    stimulus_name: str = "natural_scenes"

    # ViT logits
    vit_logits_path: str = "/home/maria/ProjectionSort/data/google_vit-base-patch16-224_embeddings_logits.pkl"

    # LFP alignment
    trial_start: float = -0.5
    trial_end: float = 0.5
    lfp_sampling_rate: float = 500.0

    # Keep every nth sample.
    # 1 -> 500 timepoints for 1 second
    # 5 -> 100 timepoints
    downsample_factor: int = 5

    # Model
    rank: int = 5
    batch_size: int = 16
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    val_size: float = 0.2
    patience: int = 15

    # Split mode:
    # "trial" = random trial holdout, same image can appear in train and val
    # "image" = hold out image identities, stricter and better for thesis validity
    split_mode: str = "image"

    # Output
    output_dir: str = "rrr_lfp_outputs"

    seed: int = 42


cfg = Config()


# -----------------------
# Reproducibility
# -----------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(cfg.seed)


# -----------------------
# Label utilities
# -----------------------

def load_vit_animate_labels(vit_logits_path: str) -> np.ndarray:
    """
    Load ViT logits and derive animate/inanimate labels.

    Returns:
        image_labels: shape (n_images,)
            1 = animate
            0 = inanimate
    """
    vit_file = np.load(vit_logits_path, allow_pickle=True)

    # Your current object:
    # vit = np.load(..., allow_pickle=True)["natural_scenes"]
    vit = vit_file["natural_scenes"]

    if vit.ndim != 2:
        raise ValueError(f"Expected ViT logits with shape (n_images, 1000), got {vit.shape}")

    top1 = np.argmax(vit, axis=1)
    image_labels = (top1 <= 397).astype(np.int64)

    print(f"Loaded ViT logits: {vit.shape}")
    print(f"Image-level label counts: {np.bincount(image_labels)}")
    print("Convention: 0 = inanimate, 1 = animate")

    return image_labels


def attach_vit_labels_to_presentations(
    presentation_table: pd.DataFrame,
    image_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Attach animate/inanimate labels to each natural-scene presentation.

    Assumes presentation_table["frame"] indexes the natural-scene image.

    Handles Allen rows where frame == -1 by dropping them, because they do not
    correspond to one of the 118 natural-scene images.
    """
    pt = presentation_table.copy()

    if "frame" not in pt.columns:
        raise ValueError(
            "presentation_table does not contain a 'frame' column.\n"
            f"Available columns: {list(pt.columns)}"
        )

    pt["frame"] = pt["frame"].astype(int)

    n_before = len(pt)

    # Allen sometimes includes frame == -1 rows. These cannot map to the 118 images.
    invalid_mask = pt["frame"] < 0
    if invalid_mask.any():
        print("\nDropping presentations with invalid frame < 0:")
        print(pt.loc[invalid_mask, ["stimulus_name", "frame", "start_time", "stop_time"]].head())
        print(f"Dropping {invalid_mask.sum()} / {len(pt)} presentations.")
        pt = pt.loc[~invalid_mask].copy()

    frames = pt["frame"].values.astype(int)

    n_images = len(image_labels)
    min_frame = frames.min()
    max_frame = frames.max()

    print(f"\nPresentation frame range after filtering: min={min_frame}, max={max_frame}")
    print(f"Number of ViT image labels: {n_images}")

    if min_frame >= 0 and max_frame < n_images:
        frame_idx = frames
        print("Using frame as 0-based index into ViT labels.")
    elif min_frame >= 1 and max_frame <= n_images:
        frame_idx = frames - 1
        print("Using frame as 1-based index; subtracting 1.")
    else:
        bad = pt.loc[(pt["frame"] < 0) | (pt["frame"] >= n_images), ["frame", "start_time", "stop_time"]]
        raise ValueError(
            "Frame values do not appear compatible with ViT label array.\n"
            f"Frame range after filtering: {min_frame}..{max_frame}; n_images={n_images}\n"
            f"Bad examples:\n{bad.head()}"
        )

    pt["image_index"] = frame_idx
    pt["label"] = image_labels[frame_idx].astype(np.int64)

    print("\nPresentation-level label counts:")
    print(pt["label"].value_counts().sort_index())

    print(f"\nKept {len(pt)} / {n_before} presentations after frame filtering.")

    return pt

# -----------------------
# LFP alignment
# -----------------------

def align_lfp_to_presentations(session, probe_id, presentation_table: pd.DataFrame):
    """
    Align LFP to natural-scene presentation onset.

    Returns xarray DataArray with dims:
        presentation_id, time_from_presentation_onset, channel
    """
    lfp = session.get_lfp(probe_id)

    presentation_times = presentation_table["start_time"].values
    presentation_ids = presentation_table.index.values

    trial_window = np.arange(
        cfg.trial_start,
        cfg.trial_end,
        1.0 / cfg.lfp_sampling_rate,
    )

    if cfg.downsample_factor > 1:
        trial_window = trial_window[::cfg.downsample_factor]

    time_selection = np.concatenate([trial_window + t for t in presentation_times])

    inds = pd.MultiIndex.from_product(
        (presentation_ids, trial_window),
        names=("presentation_id", "time_from_presentation_onset"),
    )

    ds = lfp.sel(time=time_selection, method="nearest").to_dataset(name="aligned_lfp")
    ds = ds.assign(time=inds).unstack("time")

    aligned_lfp = ds["aligned_lfp"]

    needed_dims = ("presentation_id", "time_from_presentation_onset", "channel")
    for dim in needed_dims:
        if dim not in aligned_lfp.dims:
            raise ValueError(
                f"Expected dimension {dim!r} in aligned_lfp, got dims {aligned_lfp.dims}"
            )

    aligned_lfp = aligned_lfp.transpose(*needed_dims)

    return aligned_lfp


def make_design_matrix(aligned_lfp, presentation_table_labeled: pd.DataFrame):
    """
    Convert aligned LFP into numpy arrays.

    Returns:
        X: trials x time x channels
        y: trials
        image_index: trials
        presentation_ids: trials
    """
    presentation_ids = aligned_lfp.coords["presentation_id"].values

    labeled = presentation_table_labeled.loc[presentation_ids]

    X = aligned_lfp.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = labeled["label"].values.astype(np.int64)
    image_index = labeled["image_index"].values.astype(np.int64)

    return X, y, image_index, presentation_ids


def baseline_subtract(X: np.ndarray, trial_window_values: np.ndarray) -> np.ndarray:
    """
    Subtract pre-stimulus baseline per trial/channel.

    X shape:
        trials x time x channels
    """
    baseline_mask = trial_window_values < 0

    if baseline_mask.sum() == 0:
        print("WARNING: no pre-stimulus samples found; skipping baseline subtraction.")
        return X

    baseline = X[:, baseline_mask, :].mean(axis=1, keepdims=True)
    return X - baseline


def zscore_from_train(X_train: np.ndarray, X_val: np.ndarray, eps: float = 1e-6):
    """
    Z-score per channel using only training data.

    Mean/std are computed across trials and time.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, eps)

    return (X_train - mean) / std, (X_val - mean) / std, mean, std


# -----------------------
# Splitting
# -----------------------

def make_train_val_split(y: np.ndarray, image_index: np.ndarray):
    """
    Create train/validation split.

    cfg.split_mode == "trial":
        Stratified split over presentations.
        Easier but can leak image identity across train/val.

    cfg.split_mode == "image":
        Hold out image identities.
        Better test of whether the model generalizes to unseen images.
    """
    all_trial_idx = np.arange(len(y))

    if cfg.split_mode == "trial":
        train_idx, val_idx = train_test_split(
            all_trial_idx,
            test_size=cfg.val_size,
            random_state=cfg.seed,
            stratify=y,
        )
        return train_idx, val_idx

    if cfg.split_mode == "image":
        unique_images = np.unique(image_index)

        # Label each image by its image-level label.
        image_to_label = {}
        for img in unique_images:
            labels_for_img = y[image_index == img]
            unique_labels = np.unique(labels_for_img)
            if len(unique_labels) != 1:
                raise ValueError(f"Image {img} has inconsistent labels: {unique_labels}")
            image_to_label[img] = unique_labels[0]

        image_labels = np.array([image_to_label[img] for img in unique_images])

        train_images, val_images = train_test_split(
            unique_images,
            test_size=cfg.val_size,
            random_state=cfg.seed,
            stratify=image_labels,
        )

        train_mask = np.isin(image_index, train_images)
        val_mask = np.isin(image_index, val_images)

        train_idx = all_trial_idx[train_mask]
        val_idx = all_trial_idx[val_mask]

        return train_idx, val_idx

    raise ValueError(f"Unknown split_mode: {cfg.split_mode!r}")


# -----------------------
# PyTorch dataset/model
# -----------------------

class TrialLFPTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape trials x time x channels, got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y with shape trials, got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ReducedRankLFPDecoder(nn.Module):
    """
    Single-session reduced-rank classifier.

    Input:
        x: batch x time x channel

    Parameters:
        U: channel x rank
        V: rank x time x class
        b: class

    Full decoder tensor:
        W: channel x time x class
    """

    def __init__(self, n_channels: int, n_time: int, n_classes: int = 2, rank: int = 5):
        super().__init__()

        if rank > min(n_channels, n_time):
            raise ValueError(
                f"rank={rank} must be <= min(n_channels={n_channels}, n_time={n_time})"
            )

        self.n_channels = n_channels
        self.n_time = n_time
        self.n_classes = n_classes
        self.rank = rank

        scale = 0.01
        self.U = nn.Parameter(scale * torch.randn(n_channels, rank))
        self.V = nn.Parameter(scale * torch.randn(rank, n_time, n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def weight_tensor(self):
        # channel x time x class
        return torch.einsum("nr,rtd->ntd", self.U, self.V)

    def forward(self, x):
        # x: batch x time x channel
        W = self.weight_tensor()
        logits = torch.einsum("ntd,btn->bd", W, x) + self.b
        return logits

    def latent_timecourses(self, x):
        """
        Project trials into learned rank-dimensional channel basis.

        Returns:
            batch x time x rank
        """
        return torch.einsum("btn,nr->btr", x, self.U)


# -----------------------
# Training/evaluation
# -----------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    loss_fn = nn.CrossEntropyLoss()

    losses = []
    y_all = []
    pred_all = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        pred_all.append(pred)
        y_all.append(y_batch.detach().cpu().numpy())

    y_all = np.concatenate(y_all)
    pred_all = np.concatenate(pred_all)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_all, pred_all),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    losses = []
    y_all = []
    pred_all = []
    prob_all = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        prob_animate = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()

        losses.append(loss.item())
        y_all.append(y_batch.cpu().numpy())
        pred_all.append(pred)
        prob_all.append(prob_animate)

    y_all = np.concatenate(y_all)
    pred_all = np.concatenate(pred_all)
    prob_all = np.concatenate(prob_all)

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_all, pred_all),
    }

    if len(np.unique(y_all)) == 2:
        metrics["auc"] = roc_auc_score(y_all, prob_all)
    else:
        metrics["auc"] = np.nan

    return metrics, y_all, pred_all, prob_all


# -----------------------
# Main
# -----------------------

def main():
    load_dotenv()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cache_path = os.getenv(cfg.cache_path_env)
    if cache_path is None:
        raise EnvironmentError(f"Set {cfg.cache_path_env} in your .env file.")

    manifest_path = Path(cache_path) / "manifest.json"

    print(f"Using Allen manifest: {manifest_path}")

    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

    print(f"\nLoading session {cfg.session_id}...")
    session = cache.get_session_data(cfg.session_id)

    print("\nProbe structures:")
    print({
        session.probes.loc[probe_id].description:
        list(session.channels[session.channels.probe_id == probe_id].ecephys_structure_acronym.unique())
        for probe_id in session.probes.index.values
    })

    probe_id = session.probes[
        session.probes.description == cfg.choose_probe
    ].index.values[0]

    print(f"\nUsing probe: {cfg.choose_probe}, probe_id={probe_id}")

    presentation_table = session.stimulus_presentations[
        session.stimulus_presentations.stimulus_name == cfg.stimulus_name
    ].copy()

    if len(presentation_table) == 0:
        raise ValueError(f"No presentations found for stimulus {cfg.stimulus_name!r}")

    print(f"\nNatural-scene presentations: {len(presentation_table)}")
    print("Presentation table columns:")
    print(list(presentation_table.columns))
    print(presentation_table.head())

    image_labels = load_vit_animate_labels(cfg.vit_logits_path)

    presentation_table_labeled = attach_vit_labels_to_presentations(
        presentation_table=presentation_table,
        image_labels=image_labels,
    )

    print("\nAligning LFP to image presentations...")
    aligned_lfp = align_lfp_to_presentations(
        session=session,
        probe_id=probe_id,
        presentation_table=presentation_table_labeled,
    )

    print(aligned_lfp)

    X, y, image_index, presentation_ids = make_design_matrix(
        aligned_lfp=aligned_lfp,
        presentation_table_labeled=presentation_table_labeled,
    )

    trial_window_values = aligned_lfp.coords["time_from_presentation_onset"].values
    X = baseline_subtract(X, trial_window_values)

    print(f"\nX shape: {X.shape} = trials x time x channels")
    print(f"y shape: {y.shape}")
    print(f"Presentation-level class counts: {np.bincount(y)}")
    print(f"Unique images represented: {len(np.unique(image_index))}")

    train_idx, val_idx = make_train_val_split(y, image_index)

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    X_train, X_val, z_mean, z_std = zscore_from_train(X_train, X_val)

    print("\nSplit summary:")
    print(f"split_mode = {cfg.split_mode}")
    print(f"Train X: {X_train.shape}, y counts: {np.bincount(y_train)}")
    print(f"Val   X: {X_val.shape}, y counts: {np.bincount(y_val)}")
    print(f"Train unique images: {len(np.unique(image_index[train_idx]))}")
    print(f"Val unique images:   {len(np.unique(image_index[val_idx]))}")

    train_ds = TrialLFPTensorDataset(X_train, y_train)
    val_ds = TrialLFPTensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    _, n_time, n_channels = X_train.shape

    model = ReducedRankLFPDecoder(
        n_channels=n_channels,
        n_time=n_time,
        n_classes=2,
        rank=cfg.rank,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    print("\nTraining RRR decoder...")
    for epoch in range(1, cfg.n_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={row['train_loss']:.4f}, acc={row['train_accuracy']:.3f} | "
            f"val loss={row['val_loss']:.4f}, acc={row['val_accuracy']:.3f}, auc={row['val_auc']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": cfg.__dict__,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "presentation_ids": presentation_ids,
                "image_index": image_index,
                "zscore_mean": z_mean,
                "zscore_std": z_std,
            }

            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            print(f"\nEarly stopping after {cfg.patience} non-improving epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best state recorded.")

    model.load_state_dict(best_state["model"])

    val_metrics, y_true, y_pred, y_prob = evaluate(model, val_loader, device)

    print("\nBest validation metrics:")
    print(val_metrics)

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["inanimate", "animate"],
    ))

    # Save history/predictions/model/decomposition
    history_df = pd.DataFrame(history)
    history_path = output_dir / "rrr_lfp_training_history.csv"
    history_df.to_csv(history_path, index=False)

    pred_df = pd.DataFrame({
        "presentation_id": presentation_ids[val_idx],
        "image_index": image_index[val_idx],
        "y_true": y_true,
        "y_pred": y_pred,
        "p_animate": y_prob,
    })

    pred_path = output_dir / "rrr_lfp_validation_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    model_path = output_dir / "rrr_lfp_best_model.pt"
    torch.save(best_state, model_path)

    with torch.no_grad():
        U = model.U.detach().cpu().numpy()
        V = model.V.detach().cpu().numpy()
        W = model.weight_tensor().detach().cpu().numpy()

    np.save(output_dir / "rrr_U_channel_basis.npy", U)
    np.save(output_dir / "rrr_V_temporal_basis.npy", V)
    np.save(output_dir / "rrr_W_full_weight_tensor.npy", W)

    # Save useful metadata
    meta_df = pd.DataFrame({
        "presentation_id": presentation_ids,
        "image_index": image_index,
        "label": y,
        "is_train": np.isin(np.arange(len(y)), train_idx),
        "is_val": np.isin(np.arange(len(y)), val_idx),
    })
    meta_path = output_dir / "rrr_lfp_trial_metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    print("\nSaved artifacts:")
    print(f"  {history_path}")
    print(f"  {pred_path}")
    print(f"  {model_path}")
    print(f"  {output_dir / 'rrr_U_channel_basis.npy'}")
    print(f"  {output_dir / 'rrr_V_temporal_basis.npy'}")
    print(f"  {output_dir / 'rrr_W_full_weight_tensor.npy'}")
    print(f"  {meta_path}")

    print("\nDone. RRR decomposition fitted on LFP trials.")


if __name__ == "__main__":
    main()