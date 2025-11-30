import numpy as np
from pathlib import Path
import cv2
import pyvista as pv

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
TEST_DIR = ROOT_DIR / "test_interpolation" 

# 스케일 (네가 쓰던 값 그대로)
PIXEL_SPACING_Z = 1.0      # 깊이 방향 (이미지 세로)
PIXEL_SPACING_Y = 1.0      # 폭 방향   (이미지 가로)
SLICE_SPACING_X = 0.5      # 50cm 간격
VISUAL_GAP_SCALE = 200.0     # 간격 과장용 (원하면 조절)


def load_3_masks():
    # 104, 105, 106 정렬해서 사용
    img_paths = sorted(TEST_DIR.glob("*.png"))
    if len(img_paths) < 3:
        raise RuntimeError("test_interpolation 폴더에 최소 3장 필요")

    masks = []
    for p in img_paths[:3]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없음: {p}")

        # 0~255 → 0/1 마스크
        _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32)
        masks.append(mask)
        print(f"[INFO] Loaded {p.name}, shape={mask.shape}, unique={np.unique(mask)}")

    # (z,y,x) = (세로, 가로, 슬라이스)
    vol_zyx = np.stack(masks, axis=2)
    return vol_zyx


def main():
    vol_zyx = load_3_masks()
    nz, ny, nx = vol_zyx.shape
    print(f"[INFO] Volume (z, y, x): {vol_zyx.shape}")

    p = pv.Plotter()
    p.set_background("white")

    step = 1
    print(f"[INFO] Showing every {step}-th slice")

    for k in range(0, nx, step):
        slice_zy = vol_zyx[:, :, k]

        # 지반은 밝은 회색, 공동은 진한 회색
        base = 0.8 * np.ones_like(slice_zy, dtype=np.float32)
        vis = base - 0.6 * slice_zy  # 0 -> 0.8, 1 -> 0.2

        # YZ 평면 판 하나
        img = pv.ImageData()
        img.dimensions = (1, ny, nz)  # (x, y, z) 포인트 수
        img.spacing = (1.0, PIXEL_SPACING_Y, PIXEL_SPACING_Z)

        # X 방향으로 50cm씩 밀어서 세워놓기
        x_coord = k * SLICE_SPACING_X * VISUAL_GAP_SCALE
        img.origin = (x_coord, 0.0, 0.0)

        # (z,y)를 (y,z)로 맞춰서 넣기
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

    # 인터랙티브 창 띄우면서 동시에 PNG로 저장
    p.show(screenshot=str(ROOT_DIR / "3slices_raw.png"))
    print(f"[INFO] Screenshot saved to: {ROOT_DIR / '3slices_raw.png'}")


if __name__ == "__main__":
    main()
