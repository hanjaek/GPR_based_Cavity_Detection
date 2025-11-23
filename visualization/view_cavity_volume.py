import numpy as np
from pathlib import Path
import pyvista as pv

# ================================================================
# 1) 경로 설정
# ================================================================
THIS_DIR = Path(__file__).resolve().parent
VOLUME_PATH = THIS_DIR / "cavity_volume_test.npy"

# ================================================================
# 2) 실제 공간 스케일 설정
#    - Z/Y 픽셀 간격: GPR 단면 이미지의 실제 스케일(추후 수정)
#    - X 슬라이스 간격: 장비가 이동하며 촬영한 간격 (여기선 0.5m)
# ================================================================
PIXEL_SPACING_Z = 0.05  # 이미지 세로 방향(z): 5cm 가정
PIXEL_SPACING_Y = 0.05  # 이미지 가로 방향(y): 5cm 가정
SLICE_SPACING_X = 0.5   # 슬라이스 사이 거리(x): 50cm


def main():
    # ------------------------------------------------------------
    # 3) 3D 볼륨 로드 (z, y, x)
    # ------------------------------------------------------------
    vol_zyx = np.load(VOLUME_PATH).astype(np.float32)
    nz, ny, nx = vol_zyx.shape
    print(f"[INFO] Loaded volume: {vol_zyx.shape}")

    # ------------------------------------------------------------
    # 4) PyVista는 (x, y, z) 순서 기대 → 축 재배열
    # ------------------------------------------------------------
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))

    # ------------------------------------------------------------
    # 5) PyVista UniformGrid 생성
    #    - spacing: 실제 물리적 거리 반영
    #    - origin : 좌표계 기준점
    # ------------------------------------------------------------
    grid = pv.UniformGrid()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (SLICE_SPACING_X, PIXEL_SPACING_Y, PIXEL_SPACING_Z)
    grid.origin = (0.0, 0.0, 0.0)
    grid["cavity"] = vol_xyz.ravel(order="F")

    # ------------------------------------------------------------
    # 6) iso-surface 생성
    #    - 공동(1)인 영역 경계만 3D 메쉬로 추출
    # ------------------------------------------------------------
    iso = grid.contour(isosurfaces=[0.5], scalars="cavity")

    # ------------------------------------------------------------
    # 7) 3D 시각화
    #    - 배경은 반투명 volume
    #    - 공동은 표면 메쉬(green)
    # ------------------------------------------------------------
    p = pv.Plotter()
    p.add_volume(
        grid,
        scalars="cavity",
        opacity=[0, 0, 0.1, 0.3, 0.6, 1.0],  # 지반 반투명도
        clim=[0, 1],
        shade=True,
    )
    p.add_mesh(iso, color="green", opacity=1.0)

    p.show_axes()
    p.add_bounding_box()
    p.show()


if __name__ == "__main__":
    main()
