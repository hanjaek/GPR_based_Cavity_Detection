import os
import json
import numpy as np
import pyvista as pv
from pyvista import ImageData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs")

vol_path  = os.path.join(OUT, "gpr_volume_preprocessed.npy")
meta_path = os.path.join(OUT, "gpr_volume_preprocessed_meta.json")

if not os.path.exists(vol_path):
    print("[WARN] using raw")
    vol_path  = os.path.join(OUT, "gpr_volume_norm.npy")
    meta_path = os.path.join(OUT, "gpr_volume_meta.json")

volume = np.load(vol_path)
with open(meta_path) as f:
    meta = json.load(f)

print("Loaded:", volume.shape)

S, H, W = volume.shape

# depth crop (top noise 제거)
z_min = int(H * 0.10)
z_max = int(H * 0.95)
volume = volume[:, z_min:z_max, :]
H = volume.shape[1]

# spacing per pixel
total_y = meta["spacing"]["dy_m"]
total_z = meta["spacing"]["dz_m"]
dx = meta["spacing"]["dx_m"]
dy = total_y / (W - 1)
dz = total_z / (H - 1)

nx, ny, nz = S, W, H

grid = ImageData()
grid.dimensions = (nx, ny, nz)
grid.spacing = (dx, dy, dz)
grid.origin = (0,0,0)

scalars = volume.transpose(0,2,1).ravel(order="C")
grid["Intensity"] = scalars

# better opacity curve
opacity = [0.0, 0.0, 0.05, 0.15, 0.35, 0.65, 1.0]

p = pv.Plotter()
p.add_volume(grid, scalars="Intensity",
             cmap="viridis", opacity=opacity)

p.add_axes()
p.add_bounding_box()
p.add_text("GPR Ground Volume + Internal Reflectors", font_size=12)

# aspect fix (중요!)
p.set_scale(1.0, 7.0, 12.0)
p.camera.zoom(1.8)

p.show()
