import os
import json
import numpy as np
import pyvista as pv
from pyvista import ImageData   # ★ 여기 추가

# -----------------------
# paths
# -----------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# use preprocessed volume (recommended)
vol_path  = os.path.join(OUTPUT_DIR, "gpr_volume_preprocessed.npy")
meta_path = os.path.join(OUTPUT_DIR, "gpr_volume_preprocessed_meta.json")

# fall back to raw volume if preprocessed not found
if not os.path.exists(vol_path):
    print("[WARN] preprocessed volume not found, using raw volume.")
    vol_path  = os.path.join(OUTPUT_DIR, "gpr_volume_norm.npy")
    meta_path = os.path.join(OUTPUT_DIR, "gpr_volume_meta.json")

volume = np.load(vol_path)  # (S, H, W)
with open(meta_path, "r") as f:
    meta = json.load(f)

print("Loaded volume:", volume.shape)

S, H, W = volume.shape

dx = meta["spacing"].get("dx_m", 1.0)
dy = meta["spacing"].get("dy_m", 1.0)
dz = meta["spacing"].get("dz_m", 1.0)

# -----------------------
# build pyvista ImageData (UniformGrid 대체)
# -----------------------
# Map: X -> slices, Y -> along-track, Z -> depth
nx, ny, nz = S, W, volume.shape[1]

grid = ImageData()                    # ★ 여기!
grid.dimensions = (nx, ny, nz)        # number of points in x, y, z
grid.spacing    = (dx, dy, dz)        # physical spacing
grid.origin     = (0.0, 0.0, 0.0)

# PyVista expects data flattened in C-order matching dimensions
# current volume: (S, H, W) = (X, Z, Y)
# → (X, Y, Z) 로 바꿔서 flatten
scalars = volume.transpose(0, 2, 1).ravel(order="C")
grid["Intensity"] = scalars

# -----------------------
# volume rendering
# -----------------------
pl = pv.Plotter()
pl.add_volume(
    grid,
    scalars="Intensity",
    opacity="sigmoid",     # smooth transparency
    cmap="viridis",        # color map
)

pl.add_axes(line_width=2)
pl.add_bounding_box()
pl.add_text("3D GPR Volume", font_size=12)

pl.show()
