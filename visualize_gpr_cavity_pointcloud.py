import os
import json
import numpy as np
import pyvista as pv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs")

vol = np.load(os.path.join(OUT, "gpr_volume_preprocessed.npy"))
with open(os.path.join(OUT, "gpr_volume_preprocessed_meta.json")) as f:
    meta = json.load(f)

S, H, W = vol.shape

dx = meta["spacing"]["dx_m"]
dy = meta["spacing"]["dy_m"]
dz = meta["spacing"]["dz_m"]

# --- 1) 강한 반사(99.5% 밝기)만 추출 ---
T = np.quantile(vol, 0.995)
mask = vol > T
idx = np.argwhere(mask)

if len(idx)==0:
    print("No bright points found")
    exit()

# voxel index → 실제 좌표
x = idx[:,0] * dx
y = idx[:,2] * dy
z = idx[:,1] * dz

intens = vol[mask]

# PyVista point cloud
points = np.column_stack((x, y, z))

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
p.show()
