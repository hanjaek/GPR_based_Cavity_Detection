import numpy as np
from pathlib import Path
import pyvista as pv

THIS_DIR = Path(__file__).resolve().parent
VOLUME_PATH = THIS_DIR / "cavity_volume_test1.npy"

# 도로 비율용 스페이싱 (상대 비율만 맞으면 됨)
# X: 진행 방향(도로 길이) → 길게
# Y: 도로 폭
# Z: 깊이(두께)
PIXEL_SPACING_Z = 1.0   # 깊이 방향
PIXEL_SPACING_Y = 1.0   # 폭 방향
SLICE_SPACING_X = 2.0   # 길이 방향(슬라이스 사이): 더 크게 주면 더 길쭉하게 보임(현재 테스트 데이터가 10장이라 길게 원래는 50cm = 0.5)


def main():
    # 1) 볼륨 로드 (z, y, x)
    vol_zyx = np.load(VOLUME_PATH).astype(np.float32)
    nz, ny, nx = vol_zyx.shape
    print(f"[INFO] Loaded volume (z, y, x): {vol_zyx.shape}")

    # 2) PyVista 형식으로 축 재배열 (x, y, z)
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))

    # 3) UniformGrid 생성 (도로 전체 블록)
    grid = pv.UniformGrid()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (SLICE_SPACING_X, PIXEL_SPACING_Y, PIXEL_SPACING_Z)
    grid.origin = (0.0, 0.0, 0.0)
    grid["cavity"] = vol_xyz.ravel(order="F")

    # 4) 공동 표면 iso-surface (도로 안 구멍 부분)
    iso = grid.contour(isosurfaces=[0.5], scalars="cavity")

    # 5) 도로 외곽 박스(겉 껍데기)
    bounds = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    road_box = pv.Box(bounds=bounds)

    # 6) 시각화
    p = pv.Plotter()

    # (1) 도로 껍데기: 회색, 반투명
    p.add_mesh(
        road_box,
        color="gray",
        opacity=0.35,
        smooth_shading=True,
    )

    # (2) 내부 공동: 진한 색 메쉬
    p.add_mesh(
        iso,
        color="black",     # 혹은 "green" / "red"
        opacity=1.0,
        smooth_shading=True,
    )

    # 축, 박스, 카메라 각도
    p.show_axes()
    p.add_bounding_box()

    # 카메라를 대각선 위에서 바라보는 느낌으로 세팅
    p.camera_position = "iso"   # isometric view
    p.show()


if __name__ == "__main__":
    main()
