import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 1) paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

vol_path  = os.path.join(OUTPUT_DIR, "gpr_volume_norm.npy")
meta_path = os.path.join(OUTPUT_DIR, "gpr_volume_meta.json")

volume = np.load(vol_path)          # shape: (slices, depth, length)
with open(meta_path, "r") as f:
    meta = json.load(f)

print("volume shape:", volume.shape)

# -----------------------------
# 2) simple slice visualization
# -----------------------------
S, H, W = volume.shape

# pick middle slice in X, Y, Z
mid_x = S // 2
mid_y = W // 2
mid_z = H // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) X-slice: depth vs length (original B-scan과 비슷)
axes[0].imshow(volume[mid_x, :, :], cmap="gray", aspect="auto")
axes[0].set_title("Slice along X (B-scan)")
axes[0].set_xlabel("Along-track (Y)")
axes[0].set_ylabel("Depth (Z)")

# (b) Y-slice: depth vs slice index
axes[1].imshow(volume[:, :, mid_y].T, cmap="gray", aspect="auto")
axes[1].set_title("Slice along Y")
axes[1].set_xlabel("Slice index (X)")
axes[1].set_ylabel("Depth (Z)")

# (c) Z-slice: plan view at middle depth
axes[2].imshow(volume[:, mid_z, :], cmap="gray", aspect="auto")
axes[2].set_title("Slice along Z (plan view)")
axes[2].set_xlabel("Along-track (Y)")
axes[2].set_ylabel("Slice index (X)")

plt.tight_layout()
plt.show()
