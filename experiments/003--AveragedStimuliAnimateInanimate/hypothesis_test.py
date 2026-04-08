import os
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


# ============================================================
# PATHS
# ============================================================

LFP_PATH = "/home/maria/LFPExperiments/data/mean_lfp_by_image.npy"
VIT_PATH = "/home/maria/ProjectionSort/data/google_vit-base-patch16-224_embeddings_logits.pkl"
OUT_DIR = "/home/maria/LFPExperiments/data/logreg_permutation_test"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(LFP_PATH)   # shape: (n_images, n_timepoints, n_channels)
vit = np.load(VIT_PATH, allow_pickle=True)["natural_scenes"]

top1 = np.argmax(vit, axis=1)
y = (top1 <= 397).astype(np.int64)   # 1 = animate, 0 = inanimate

print("X shape:", X.shape)
print("y shape:", y.shape)

n_images, n_timepoints, n_channels = X.shape
X_flat = X.reshape(n_images, n_timepoints * n_channels)

print("Flattened X shape:", X_flat.shape)
print("Animate count:", y.sum(), "Inanimate count:", len(y) - y.sum())


# ============================================================
# CV PIPELINE
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42
N_PERMUTATIONS = 1000   # start with 200 for speed, then increase to 1000+

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])


# ============================================================
# HELPER: GET OOF SCORES FOR A GIVEN LABEL VECTOR
# ============================================================

def cross_validated_scores(X, y, cv, clf):
    """
    Returns out-of-fold predictions and summary metrics.
    """
    oof_pred = np.zeros(len(y), dtype=np.int64)
    oof_prob = np.zeros(len(y), dtype=np.float64)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        clf.fit(X_train, y_train)

        oof_pred[test_idx] = clf.predict(X_test)
        oof_prob[test_idx] = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y, oof_pred)
    bal_acc = balanced_accuracy_score(y, oof_pred)
    auc = roc_auc_score(y, oof_prob)

    return {
        "oof_pred": oof_pred,
        "oof_prob": oof_prob,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc": auc,
    }


# ============================================================
# OBSERVED SCORE
# ============================================================

observed = cross_validated_scores(X_flat, y, cv, clf)

print("\nObserved results")
print(f"Accuracy:          {observed['accuracy']:.4f}")
print(f"Balanced accuracy: {observed['balanced_accuracy']:.4f}")
print(f"ROC AUC:           {observed['auc']:.4f}")


# ============================================================
# PERMUTATION NULL
# ============================================================

rng = np.random.default_rng(12345)

perm_acc = np.zeros(N_PERMUTATIONS)
perm_bal_acc = np.zeros(N_PERMUTATIONS)
perm_auc = np.zeros(N_PERMUTATIONS)

for b in range(N_PERMUTATIONS):
    y_perm = rng.permutation(y)

    res = cross_validated_scores(X_flat, y_perm, cv, clf)

    perm_acc[b] = res["accuracy"]
    perm_bal_acc[b] = res["balanced_accuracy"]
    perm_auc[b] = res["auc"]

    if (b + 1) % 50 == 0:
        print(f"Done {b+1}/{N_PERMUTATIONS} permutations")


# ============================================================
# P-VALUES
# ============================================================

p_acc = (1 + np.sum(perm_acc >= observed["accuracy"])) / (1 + N_PERMUTATIONS)
p_bal_acc = (1 + np.sum(perm_bal_acc >= observed["balanced_accuracy"])) / (1 + N_PERMUTATIONS)
p_auc = (1 + np.sum(perm_auc >= observed["auc"])) / (1 + N_PERMUTATIONS)

print("\nPermutation test results")
print(f"p-value (accuracy):          {p_acc:.4f}")
print(f"p-value (balanced accuracy): {p_bal_acc:.4f}")
print(f"p-value (ROC AUC):           {p_auc:.4f}")


# ============================================================
# SAVE
# ============================================================

np.save(os.path.join(OUT_DIR, "perm_acc.npy"), perm_acc)
np.save(os.path.join(OUT_DIR, "perm_bal_acc.npy"), perm_bal_acc)
np.save(os.path.join(OUT_DIR, "perm_auc.npy"), perm_auc)

np.save(os.path.join(OUT_DIR, "observed_oof_pred.npy"), observed["oof_pred"])
np.save(os.path.join(OUT_DIR, "observed_oof_prob.npy"), observed["oof_prob"])
np.save(os.path.join(OUT_DIR, "y_true.npy"), y)

with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(f"Observed accuracy: {observed['accuracy']:.6f}\n")
    f.write(f"Observed balanced_accuracy: {observed['balanced_accuracy']:.6f}\n")
    f.write(f"Observed auc: {observed['auc']:.6f}\n")
    f.write(f"p_acc: {p_acc:.6f}\n")
    f.write(f"p_bal_acc: {p_bal_acc:.6f}\n")
    f.write(f"p_auc: {p_auc:.6f}\n")

print(f"\nSaved outputs to: {OUT_DIR}")