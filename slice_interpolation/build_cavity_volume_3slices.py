import numpy as np
from pathlib import Path
import cv2

from sdt_interpolation import build_volume_from_masks

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
TEST_DIR = ROOT_DIR / "test_interpolation"

OUT_RAW = THIS_DIR / "cavity_volume_3slices_raw.npy"
OUT_INTERP = THIS_DIR / "cavity_volume_3slices_interp.npy"


def load_masks_3():
    img_paths = sorted(TEST_DIR.glob("*.png"))
    if len(img_paths) < 3:
        raise RuntimeError("test_interpolation 폴더에 최소 3장 필요")

    masks = []
    for p in img_paths[:3]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없음: {p}")
        _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        masks.append(mask)
        print(f"[INFO] Loaded {p.name}, shape={mask.shape}, unique={np.unique(mask)}")

    return masks


def main():
    masks_zy = load_masks_3()
    nz, ny = masks_zy[0].shape

    # 1) 원본 3장 그대로 (발표 1단계용)
    vol_raw = np.zeros((nz, ny, len(masks_zy)), dtype=np.uint8)
    for i, m in enumerate(masks_zy):
        vol_raw[:, :, i] = m
    np.save(OUT_RAW, vol_raw)
    print(f"[INFO] Saved RAW volume: {OUT_RAW}, shape={vol_raw.shape}")

    # 2) SDT 보간된 볼륨 (발표 2단계용)
    vol_interp, num_mid = build_volume_from_masks(
        masks_zy,
        orig_spacing_x=0.5,   # 50cm
        target_spacing_x=0.02  # 10cm 간격으로 보간
    )
    np.save(OUT_INTERP, vol_interp)
    print(f"[INFO] Saved INTERP volume: {OUT_INTERP}, shape={vol_interp.shape}")
    print(f"[INFO] num_mid between slices: {num_mid}")


if __name__ == "__main__":
    main()
