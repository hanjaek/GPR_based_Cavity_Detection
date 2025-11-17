import os
import json
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs")

# load preprocessed data
vol = np.load(os.path.join(OUT, "gpr_volume_preprocessed.npy"))
with open(os.path.join(OUT, "gpr_volume_preprocessed_meta.json")) as f:
    meta = json.load(f)

S, H, W = vol.shape
dx = meta["spacing"]["dx_m"]
dy = meta["spacing"]["dy_m"]
dz = meta["spacing"]["dz_m"]

print("Volume:", vol.shape)

# --- 1) smoothing 먼저 (노이즈 제거)
vol_smooth = gaussian_filter(vol, sigma=1.0)

# --- 2) bright threshold (상위 1.5%만)
T = np.quantile(vol_smooth, 0.985)
mask = vol_smooth > T

idx = np.argwhere(mask)
if len(idx) == 0:
    print("No bright points found")
    exit()

# --- 3) correct XYZ mapping ---
X = idx[:, 0] * dx                     # 슬라이스 index
Y = (idx[:, 2] / (W - 1)) * dy * W     # 진행 방향 보정
Z = (idx[:, 1] / (H - 1)) * dz * H     # 깊이 보정 (scale 문제 해결)

intens = vol_smooth[mask]

# --- 4) PyVista point cloud ---
points = np.column_stack((X, Y, Z))
cloud = pv.PolyData(points)
cloud["intensity"] = intens

p = pv.Plotter()
p.add_points(
    cloud,
    scalars="intensity",
    render_points_as_spheres=True,
    point_size=5,
    cmap="viridis",
)

p.add_axes()
p.add_bounding_box()
p.add_text("GPR Cavity Extracted 3D Cloud", font_size=12)
p.show()
