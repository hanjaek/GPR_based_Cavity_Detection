import numpy as np
from pathlib import Path
import pyvista as pv

THIS_DIR = Path(__file__).resolve().parent
VOLUME_PATH = THIS_DIR / "cavity_volume_3slices_interp.npy"

PIXEL_SPACING_Z = 1.0
PIXEL_SPACING_Y = 1.0

# 🔥 첫 번째 그림에서 쓰던 전체 길이(대충 200) 그대로 쓰자
DESIRED_TOTAL_X = 200.0   # 첫 번째 3장 그림과 동일한 범위


def main():
    vol_zyx = np.load(VOLUME_PATH).astype(np.float32)  # (z, y, x)
    nz, ny, nx = vol_zyx.shape
    print(f"[INFO] Loaded interp volume (z,y,x): {vol_zyx.shape}")

    # 🔥 슬라이스 개수(nx)에 맞춰 0 ~ DESIRED_TOTAL_X 사이를 균등 분할
    if nx > 1:
        step_x = DESIRED_TOTAL_X / (nx - 1)
    else:
        step_x = 0.0

    p = pv.Plotter()
    p.set_background("white")

    for k in range(nx):
        slice_zy = vol_zyx[:, :, k]

        base = 0.8 * np.ones_like(slice_zy, dtype=np.float32)
        vis = base - 0.6 * slice_zy

        img = pv.ImageData()
        img.dimensions = (1, ny, nz)      # (x, y, z)
        img.spacing = (1.0, PIXEL_SPACING_Y, PIXEL_SPACING_Z)

        # 🔥 여기! k=0 ~ nx-1이 0 ~ DESIRED_TOTAL_X 사이에 고정되도록
        x_coord = k * step_x
        img.origin = (x_coord, 0, 0)

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
    p.camera_position = "iso"
    p.show()


if __name__ == "__main__":
    main()
