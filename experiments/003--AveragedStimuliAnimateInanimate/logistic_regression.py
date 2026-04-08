import os
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

LFP_PATH = "/home/maria/LFPExperiments/data/mean_lfp_by_image.npy"
VIT_PATH = "/home/maria/ProjectionSort/data/google_vit-base-patch16-224_embeddings_logits.pkl"
OUT_DIR = "/home/maria/LFPExperiments/data/logreg_cv_results"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(LFP_PATH)   # expected shape: (n_images, n_timepoints, n_channels)
print("Loaded LFP shape:", X.shape)

vit = np.load(VIT_PATH, allow_pickle=True)["natural_scenes"]   # (images, 1000)
top1 = np.argmax(vit, axis=1)
y = (top1 <= 397).astype(np.int64)   # 1 = animate, 0 = inanimate

print("Loaded labels shape:", y.shape)
print("Animate count:", y.sum())
print("Inanimate count:", len(y) - y.sum())

if X.shape[0] != len(y):
    raise ValueError(
        f"Mismatch: X has {X.shape[0]} samples but y has {len(y)} labels"
    )

if X.ndim != 3:
    raise ValueError(
        f"Expected X to have shape (n_images, n_timepoints, n_channels), got {X.shape}"
    )


# ============================================================
# FLATTEN
# ============================================================

n_images, n_timepoints, n_channels = X.shape
X_flat = X.reshape(n_images, n_timepoints * n_channels)

print("Flattened shape:", X_flat.shape)


# ============================================================
# CROSS-VALIDATION SETUP
# ============================================================

n_splits = 5
random_state = 42

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

# Standardize inside each fold, then fit logistic regression
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",   # good for smaller datasets
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state
    ))
])


# ============================================================
# OUT-OF-FOLD STORAGE
# ============================================================

oof_pred = np.zeros(len(y), dtype=np.int64)
oof_prob = np.zeros(len(y), dtype=np.float64)

fold_metrics = []


# ============================================================
# RUN CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(cv.split(X_flat, y), start=1):
    X_train, X_test = X_flat[train_idx], X_flat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    oof_pred[test_idx] = y_pred
    oof_prob[test_idx] = y_prob

    fold_acc = accuracy_score(y_test, y_pred)
    fold_bal_acc = balanced_accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) == 2:
        fold_auc = roc_auc_score(y_test, y_prob)
    else:
        fold_auc = np.nan

    fold_metrics.append({
        "fold": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": fold_acc,
        "balanced_accuracy": fold_bal_acc,
        "roc_auc": fold_auc
    })

    print(f"\nFold {fold}")
    print(f"  train size: {len(train_idx)}")
    print(f"  test size:  {len(test_idx)}")
    print(f"  accuracy:   {fold_acc:.4f}")
    print(f"  bal acc:    {fold_bal_acc:.4f}")
    print(f"  roc auc:    {fold_auc:.4f}")


# ============================================================
# FINAL OOF METRICS
# ============================================================

overall_acc = accuracy_score(y, oof_pred)
overall_bal_acc = balanced_accuracy_score(y, oof_pred)
overall_auc = roc_auc_score(y, oof_prob)

print("\n" + "=" * 60)
print("OVERALL OUT-OF-FOLD RESULTS")
print("=" * 60)
print(f"Accuracy:          {overall_acc:.4f}")
print(f"Balanced accuracy: {overall_bal_acc:.4f}")
print(f"ROC AUC:           {overall_auc:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y, oof_pred))

print("\nClassification report:")
print(classification_report(y, oof_pred, digits=4))


# ============================================================
# SAVE RESULTS
# ============================================================

np.save(os.path.join(OUT_DIR, "oof_pred.npy"), oof_pred)
np.save(os.path.join(OUT_DIR, "oof_prob.npy"), oof_prob)
np.save(os.path.join(OUT_DIR, "y_true.npy"), y)

# Save fold metrics in a simple structured way
with open(os.path.join(OUT_DIR, "fold_metrics.txt"), "w") as f:
    for row in fold_metrics:
        f.write(str(row) + "\n")
    f.write("\n")
    f.write(f"Overall accuracy: {overall_acc:.6f}\n")
    f.write(f"Overall balanced accuracy: {overall_bal_acc:.6f}\n")
    f.write(f"Overall ROC AUC: {overall_auc:.6f}\n")

print(f"\nSaved results to: {OUT_DIR}")