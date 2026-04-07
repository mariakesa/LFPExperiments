import os
import math
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load tensor
# ----------------------------
path = "/home/maria/LFPExperiments/data/mean_lfp_by_image.npy"
x = np.load(path)

print("Loaded shape:", x.shape)

if x.ndim != 3:
    raise ValueError(f"Expected a 3D array (n_images, n_channels, n_timepoints), got shape {x.shape}")

n_images, n_channels, n_timepoints = x.shape

# ----------------------------
# Global color scale
# ----------------------------
vabs = np.max(np.abs(x))
vmin, vmax = -vabs, vabs

# ----------------------------
# Grid layout
# ----------------------------
ncols = 10
nrows = math.ceil(n_images / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(2.2 * ncols, 2.2 * nrows),
    squeeze=False
)

# ----------------------------
# Plot each image fingerprint
# ----------------------------
for i in range(n_images):
    r = i // ncols
    c = i % ncols
    ax = axes[r, c]

    im = ax.imshow(
        x[i],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(f"Image {i}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

# Hide unused subplots
for j in range(n_images, nrows * ncols):
    r = j // ncols
    c = j % ncols
    axes[r, c].axis("off")

# Shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
cbar.set_label("LFP amplitude")

fig.suptitle("Mean LFP fingerprint for each image", fontsize=16)
plt.tight_layout()
plt.show()