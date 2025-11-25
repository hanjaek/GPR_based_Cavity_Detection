import numpy as np
from pathlib import Path
import pyvista as pv

THIS_DIR = Path(__file__).resolve().parent
VOLUME_PATH = THIS_DIR / "cavity_volume.npy"

# 스케일
PIXEL_SPACING_Z = 1.0      # 깊이 방향 (이미지 세로)
PIXEL_SPACING_Y = 1.0      # 폭 방향   (이미지 가로)
SLICE_SPACING_X = 0.5       # 슬라이스 간 거리 50cm
VISUAL_GAP_SCALE = 4.0      # 간격 과장하고 싶으면 2.0, 3.0 등으로


def main():
    # vol_zyx: (z, y, x)
    vol_zyx = np.load(VOLUME_PATH).astype(np.float32)
    nz, ny, nx = vol_zyx.shape
    print(f"[INFO] Loaded volume (z, y, x): {vol_zyx.shape}")

    p = pv.Plotter()
    p.set_background("white")

    # 너무 많으면 보기 힘드니까 간격 샘플링 (원하면 step=1로 바꿔도 됨)
    # step = max(nx // 30, 1)
    step = 1
    print(f"[INFO] Showing every {step}-th slice")

    for k in range(0, nx, step):
        # 한 장: (z, y) = (깊이, 폭)
        slice_zy = vol_zyx[:, :, k]

        # 지반은 밝은 회색, 공동은 진한 회색
        base = 0.8 * np.ones_like(slice_zy, dtype=np.float32)
        vis = base - 0.6 * slice_zy  # 0 -> 0.8, 1 -> 0.2

        # 여기서 만드는 판은 "YZ 평면"에 세워질 거야.
        img = pv.ImageData()

        # nx, ny, nz = (x, y, z) 포인트수
        # x는 1칸(두께 거의 0), y/z는 이미지 크기
        img.dimensions = (1, ny, nz)
        img.spacing = (1.0, PIXEL_SPACING_Y, PIXEL_SPACING_Z)

        # X 방향으로 50cm씩 밀어서 세워놓기
        x_coord = k * SLICE_SPACING_X * VISUAL_GAP_SCALE
        img.origin = (x_coord, 0.0, 0.0)

        # (z, y)를 (y, z)로 맞춰서 넣어줌
        img["val"] = vis.T.ravel(order="F")

        p.add_mesh(
            img,
            scalars="val",
            cmap="gray",
            opacity=0.5,
            show_scalar_bar=False,
        )

    p.show_axes()
    p.add_bounding_box()

    # 대각선에서 보는 뷰 (첫 번째 그림 느낌)
    p.camera_position = "iso"
    p.show()


if __name__ == "__main__":
    main()