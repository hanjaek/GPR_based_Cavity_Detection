import numpy as np
from pathlib import Path
import cv2
from .sdt_interpolation import build_volume_from_masks

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent     # project_root/
TEST_DIR = ROOT_DIR / "test_interpolation"

OUT_RAW = THIS_DIR / "cavity_volume_3slices_raw.npy"
OUT_INTERP = THIS_DIR / "cavity_volume_3slices_interp.npy"


def load_masks_3():
    """
    test_interpolation 폴더 안에 있는 3장을 로드.
    파일명 정렬 순서대로 사용.
    """
    img_paths = sorted(TEST_DIR.glob("*"))
    assert len(img_paths) >= 3, "test_interpolation 폴더에 최소 3장 넣어줘!"

    masks = []
    for p in img_paths[:3]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        # 마스크가 이미 0/1이면 그대로 쓰고,
        # 아니면 threshold로 바이너리화
        _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        masks.append(mask.astype(np.uint8))

        print(f"[INFO] Loaded {p.name}, shape={mask.shape}")

    return masks


def main():
    masks_zy = load_masks_3()
    nz, ny = masks_zy[0].shape

    # -----------------------------
    # (1) 원본 3장 그대로 볼륨 생성
    # -----------------------------
    nx_raw = 3
    vol_raw = np.zeros((nz, ny, nx_raw), dtype=np.uint8)
    for i, m in enumerate(masks_zy):
        vol_raw[:, :, i] = m

    np.save(OUT_RAW, vol_raw)
    print(f"[INFO] Saved RAW 3-slice volume: {OUT_RAW}, shape={vol_raw.shape}")

    # -----------------------------
    # (2) SDT 보간 → 이어진 공동 볼륨 생성
    # -----------------------------
    vol_interp, num_mid = build_volume_from_masks(
        masks_zy,
        orig_spacing_x=0.5,   # 원본 50cm
        target_spacing_x=0.1  # 10cm 단위 보간
    )

    np.save(OUT_INTERP, vol_interp)
    print(f"[INFO] Saved INTERP volume: {OUT_INTERP}, shape={vol_interp.shape}")
    print(f"[INFO] num_mid between slices: {num_mid}")


if __name__ == "__main__":
    main()