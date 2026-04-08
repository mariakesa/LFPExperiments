import os
import copy
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    lfp_path: str = "/home/maria/LFPExperiments/data/mean_lfp_by_image.npy"
    vit_logits_path: str = "/home/maria/ProjectionSort/data/google_vit-base-patch16-224_embeddings_logits.pkl"
    out_dir: str = "/home/maria/LFPExperiments/data/resnet18_cv_results"

    n_splits: int = 5
    random_state: int = 42

    batch_size: int = 8
    num_workers: int = 4

    image_size: int = 224
    epochs: int = 15
    unfreeze_after_epoch: int = 5

    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)


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

X = np.load(cfg.lfp_path)   # expected: (n_images, n_timepoints, n_channels)
vit = np.load(cfg.vit_logits_path, allow_pickle=True)["natural_scenes"]

top1 = np.argmax(vit, axis=1)
y = (top1 <= 397).astype(np.int64)   # 1 = animate, 0 = inanimate

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Animate:", y.sum(), "Inanimate:", len(y) - y.sum())

if X.shape[0] != len(y):
    raise ValueError(f"Mismatch: X has {X.shape[0]} samples but y has {len(y)} labels")

if X.ndim != 3:
    raise ValueError(f"Expected X to have shape (n_images, n_timepoints, n_channels), got {X.shape}")


# ============================================================
# DATASET
# ============================================================

class LFPImageDataset(Dataset):
    def __init__(self, X, y, indices, transform=None):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def _to_pil(self, arr_2d: np.ndarray) -> Image.Image:
        # arr_2d shape: (timepoints, channels)
        arr = arr_2d.astype(np.float32)

        # robust per-sample normalization
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            lo, hi = arr.min(), arr.max()

        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        arr_uint8 = (255.0 * arr).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="L")

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i]
        label = int(self.y[i])

        img = self._to_pil(x)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), int(i)


weights = ResNet18_Weights.DEFAULT
mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = resnet18(weights=weights)

    # replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    # freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


def make_optimizer(model):
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


criterion = nn.CrossEntropyLoss()


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    all_probs, all_preds, all_targets = [], [], []

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

    return {
        "loss": running_loss / len(loader.dataset),
        "acc": accuracy_score(all_targets, all_preds),
        "bal_acc": balanced_accuracy_score(all_targets, all_preds),
        "auc": roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) == 2 else np.nan,
    }


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_preds, all_targets, all_ids = [], [], [], []

    for xb, yb, ids in loader:
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
        all_ids.append(np.asarray(ids))

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


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=cfg.n_splits,
    shuffle=True,
    random_state=cfg.random_state
)

oof_probs = np.zeros(len(y), dtype=np.float32)
oof_preds = np.zeros(len(y), dtype=np.int64)
oof_targets = y.copy()

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

    train_ds = LFPImageDataset(X, y, train_idx, transform=train_transform)
    val_ds = LFPImageDataset(X, y, val_idx, transform=val_transform)

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

    best_state = None
    best_bal_acc = -np.inf

    for epoch in range(cfg.epochs):
        if epoch == cfg.unfreeze_after_epoch:
            print("Unfreezing backbone")
            unfreeze_backbone(model)
            optimizer = make_optimizer(model)

        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg.device)
        val_metrics, _, _, _, _ = eval_one_epoch(model, val_loader, cfg.device)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} "
            f"bal_acc={train_metrics['bal_acc']:.3f} auc={train_metrics['auc']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} "
            f"bal_acc={val_metrics['bal_acc']:.3f} auc={val_metrics['auc']:.3f}"
        )

        if val_metrics["bal_acc"] > best_bal_acc:
            best_bal_acc = val_metrics["bal_acc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_metrics, val_ids, val_targets, val_preds, val_probs = eval_one_epoch(
        model, val_loader, cfg.device
    )

    oof_probs[val_ids] = val_probs
    oof_preds[val_ids] = val_preds

    fold_metrics.append({
        "fold": fold,
        "acc": val_metrics["acc"],
        "bal_acc": val_metrics["bal_acc"],
        "auc": val_metrics["auc"],
    })

    print(
        f"Best fold {fold}: "
        f"acc={val_metrics['acc']:.3f}, "
        f"bal_acc={val_metrics['bal_acc']:.3f}, "
        f"auc={val_metrics['auc']:.3f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

overall_acc = accuracy_score(oof_targets, oof_preds)
overall_bal_acc = balanced_accuracy_score(oof_targets, oof_preds)
overall_auc = roc_auc_score(oof_targets, oof_probs)

print("\n========== OOF CV RESULTS ==========")
print(f"OOF accuracy:          {overall_acc:.4f}")
print(f"OOF balanced accuracy: {overall_bal_acc:.4f}")
print(f"OOF ROC AUC:           {overall_auc:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(oof_targets, oof_preds))

print("\nClassification report:")
print(classification_report(oof_targets, oof_preds, digits=4))

np.save(os.path.join(cfg.out_dir, "oof_probs.npy"), oof_probs)
np.save(os.path.join(cfg.out_dir, "oof_preds.npy"), oof_preds)
np.save(os.path.join(cfg.out_dir, "oof_targets.npy"), oof_targets)

with open(os.path.join(cfg.out_dir, "fold_metrics.txt"), "w") as f:
    for row in fold_metrics:
        f.write(str(row) + "\n")
    f.write("\n")
    f.write(f"Overall accuracy: {overall_acc:.6f}\n")
    f.write(f"Overall balanced accuracy: {overall_bal_acc:.6f}\n")
    f.write(f"Overall ROC AUC: {overall_auc:.6f}\n")

print(f"\nSaved outputs to: {cfg.out_dir}")