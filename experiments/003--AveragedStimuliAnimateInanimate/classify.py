import os
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    lfp_path: str = "/home/maria/LFPExperiments/data/mean_lfp_by_image.npy"
    vit_logits_path: str = "/home/maria/ProjectionSort/data/google_vit-base-patch16-224_embeddings_logits.pkl"
    output_dir: str = "/home/maria/LFPExperiments/data/vit_cv_results"

    n_splits: int = 5
    random_state: int = 42

    image_size: int = 224
    batch_size: int = 8
    num_workers: int = 4

    epochs: int = 20
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    unfreeze_backbone: bool = True
    unfreeze_after_epoch: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.random_state)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(cfg.lfp_path)  # expected shape: (n_images, n_timepoints, n_channels)
print("Loaded LFP shape:", X.shape)

vit = np.load(cfg.vit_logits_path, allow_pickle=True)["natural_scenes"]  # (n_images, 1000)
top1 = np.argmax(vit, axis=1)
y = (top1 <= 397).astype(np.int64)  # 1 = animate, 0 = inanimate

print("Loaded label shape:", y.shape)
print("Animate count:", y.sum(), "Inanimate count:", len(y) - y.sum())

if X.shape[0] != len(y):
    raise ValueError(
        f"Mismatch: LFP has {X.shape[0]} samples but labels have {len(y)} samples."
    )

if X.ndim != 3:
    raise ValueError(f"Expected X to have shape (n_images, n_timepoints, n_channels), got {X.shape}")


# ============================================================
# DATASET
# ============================================================

class LFPFingerprintDataset(Dataset):
    def __init__(self, X, y, indices, transform=None):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def _to_uint8_image(self, arr_2d: np.ndarray) -> Image.Image:
        # arr_2d shape: (timepoints, channels)
        arr = arr_2d.astype(np.float32)

        # Per-sample robust normalization
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            lo, hi = arr.min(), arr.max()

        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        arr_uint8 = (255.0 * arr).astype(np.uint8)

        # PIL expects H x W
        return Image.fromarray(arr_uint8, mode="L")

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i]              # (timepoints, channels)
        label = int(self.y[i])

        img = self._to_uint8_image(x)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), int(i)


weights = ViT_B_16_Weights.DEFAULT
imagenet_mean = weights.transforms().mean
imagenet_std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 2)

    if cfg.unfreeze_backbone:
        # Start frozen, then unfreeze later
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.heads.head.parameters():
            param.requires_grad = True

    return model


def unfreeze_model_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * xb.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "acc": accuracy_score(all_targets, all_preds),
        "bal_acc": balanced_accuracy_score(all_targets, all_preds),
        "auc": roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) == 2 else np.nan,
    }
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []
    all_ids = []

    for xb, yb, sample_ids in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * xb.size(0)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
        all_ids.append(np.array(sample_ids))

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_ids = np.concatenate(all_ids)

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "acc": accuracy_score(all_targets, all_preds),
        "bal_acc": balanced_accuracy_score(all_targets, all_preds),
        "auc": roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) == 2 else np.nan,
    }

    return metrics, all_ids, all_targets, all_preds, all_probs


def make_optimizer(model):
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "heads.head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": cfg.lr_backbone,
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": cfg.lr_head,
        })

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


# ============================================================
# CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=cfg.n_splits,
    shuffle=True,
    random_state=cfg.random_state
)

criterion = nn.CrossEntropyLoss()

oof_probs = np.zeros(len(y), dtype=np.float32)
oof_preds = np.zeros(len(y), dtype=np.int64)
oof_targets = y.copy()

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

    train_ds = LFPFingerprintDataset(X, y, train_idx, transform=train_transform)
    val_ds = LFPFingerprintDataset(X, y, val_idx, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    model = build_model().to(cfg.device)
    optimizer = make_optimizer(model)

    best_model_state = None
    best_val_bal_acc = -np.inf

    for epoch in range(cfg.epochs):
        if cfg.unfreeze_backbone and epoch == cfg.unfreeze_after_epoch:
            print("Unfreezing backbone")
            unfreeze_model_backbone(model)
            optimizer = make_optimizer(model)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_metrics, _, _, _, _ = eval_one_epoch(model, val_loader, criterion, cfg.device)

        print(
            f"Epoch {epoch+1:02d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} bal_acc={train_metrics['bal_acc']:.3f} auc={train_metrics['auc']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} bal_acc={val_metrics['bal_acc']:.3f} auc={val_metrics['auc']:.3f}"
        )

        if val_metrics["bal_acc"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["bal_acc"]
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    val_metrics, val_ids, val_targets, val_preds, val_probs = eval_one_epoch(
        model, val_loader, criterion, cfg.device
    )

    oof_probs[val_ids] = val_probs
    oof_preds[val_ids] = val_preds

    fold_result = {
        "fold": fold,
        "acc": val_metrics["acc"],
        "bal_acc": val_metrics["bal_acc"],
        "auc": val_metrics["auc"],
    }
    fold_metrics.append(fold_result)

    print(
        f"Best fold {fold}: "
        f"acc={val_metrics['acc']:.3f}, "
        f"bal_acc={val_metrics['bal_acc']:.3f}, "
        f"auc={val_metrics['auc']:.3f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

cv_acc = accuracy_score(oof_targets, oof_preds)
cv_bal_acc = balanced_accuracy_score(oof_targets, oof_preds)
cv_auc = roc_auc_score(oof_targets, oof_probs)

print("\n========== OOF CV RESULTS ==========")
print(f"OOF accuracy:          {cv_acc:.4f}")
print(f"OOF balanced accuracy: {cv_bal_acc:.4f}")
print(f"OOF ROC AUC:           {cv_auc:.4f}")

fold_df = []
for row in fold_metrics:
    fold_df.append([row["fold"], row["acc"], row["bal_acc"], row["auc"]])

fold_df = np.array(fold_df, dtype=object)
print("\nPer-fold metrics:")
for row in fold_metrics:
    print(row)

np.save(os.path.join(cfg.output_dir, "oof_probs.npy"), oof_probs)
np.save(os.path.join(cfg.output_dir, "oof_preds.npy"), oof_preds)
np.save(os.path.join(cfg.output_dir, "oof_targets.npy"), oof_targets)
np.save(os.path.join(cfg.output_dir, "fold_metrics.npy"), fold_df)

print(f"\nSaved outputs to: {cfg.output_dir}")