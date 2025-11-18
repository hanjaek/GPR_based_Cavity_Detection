import os
import json
import numpy as np
import pyvista as pv
from pyvista import ImageData

# -----------------------
# 1) 경로 설정
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs")

# 우선 preprocessed가 있으면 사용, 없으면 raw 사용
vol_path  = os.path.join(OUT, "gpr_volume_preprocessed.npy")
meta_path = os.path.join(OUT, "gpr_volume_preprocessed_meta.json")

if not os.path.exists(vol_path):
    print("[WARN] using raw volume")
    vol_path  = os.path.join(OUT, "gpr_volume_norm.npy")
    meta_path = os.path.join(OUT, "gpr_volume_meta.json")

volume = np.load(vol_path)       # shape: (S, H, W)
with open(meta_path, "r") as f:
    meta = json.load(f)

print("Loaded volume:", volume.shape)
S, H, W = volume.shape

# -----------------------
# 2) 스케일 설정 (meta 기반, 픽셀당 간격으로 변환)
# -----------------------
dx_total = meta["spacing"]["dx_m"]          # 슬라이스 간격(이미 per-slice)
dy_total = meta["spacing"]["dy_m"]          # 전체 진행 방향 길이 (예: 10m)
dz_total = meta["spacing"]["dz_m"]          # 전체 깊이 (예: 5m)

dx = dx_total                               # 한 슬라이스 사이 거리
dy = dy_total / max(W - 1, 1)               # 진행 방향 픽셀당 거리
dz = dz_total / max(H - 1, 1)               # 깊이 픽셀당 거리

print(f"Spacing per pixel (m): dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

# -----------------------
# 3) PyVista ImageData 생성
# -----------------------
# PyVista dimension 순서: (nx, ny, nz) = (X, Y, Z)
nx, ny, nz = S, W, H

grid = ImageData()
grid.dimensions = (nx, ny, nz)
grid.spacing    = (dx, dy, dz)
grid.origin     = (0.0, 0.0, 0.0)

# volume: (S, H, W) = (X, Z, Y)
# → (X, Y, Z) 순서로 바꾸고 flatten
scalars = volume.transpose(0, 2, 1).ravel(order="C")
grid["Intensity"] = scalars

# -----------------------
# 4) 볼륨 렌더링 (지반 + 내부 반사체)
# -----------------------
# 어두운 값은 거의 투명, 밝은 값은 진하게
opacity = [0.0, 0.0, 0.05, 0.2, 0.4, 0.7, 1.0]

p = pv.Plotter()
p.add_volume(
    grid,
    scalars="Intensity",
    cmap="viridis",
    opacity=opacity,
)

p.add_axes()
p.add_bounding_box()
p.add_text("3D GPR Ground Volume", font_size=12)

# 축 비율은 일단 기본값(물리 스케일 그대로) 사용
p.show()
