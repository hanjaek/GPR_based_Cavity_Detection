import numpy as np
from pathlib import Path
from .sdt_interpolation import build_volume_from_masks

THIS_DIR = Path(__file__).resolve().parent

OUT_RAW = THIS_DIR / "cavity_volume_3slices_raw.npy"
OUT_INTERP = THIS_DIR / "cavity_volume_3slices_interp.npy"


def load_masks_3():
    """
    TODO: 여기 부분을 실제 마스크 경로에 맞게 수정해줘.
    현재는 예시로 mask0.npy, mask1.npy, mask2.npy 사용.
    """
    m0 = np.load(THIS_DIR / "mask0.npy")  # shape: (z, y), 값: 0/1
    m1 = np.load(THIS_DIR / "mask1.npy")
    m2 = np.load(THIS_DIR / "mask2.npy")
    return [m0, m1, m2]


def main():
    masks_zy = load_masks_3()
    nz, ny = masks_zy[0].shape

    # -----------------------------
    # 1) 원본 3장만 쌓은 볼륨
    # -----------------------------
    nx_raw = len(masks_zy)
    vol_raw = np.zeros((nz, ny, nx_raw), dtype=np.uint8)
    for i, m in enumerate(masks_zy):
        vol_raw[:, :, i] = m

    np.save(OUT_RAW, vol_raw)
    print(f"[INFO] Saved RAW 3-slice volume: {OUT_RAW}, shape={vol_raw.shape}")

    # -----------------------------
    # 2) SDT 보간 적용한 이어진 공동 볼륨
    # -----------------------------
    vol_interp, num_mid = build_volume_from_masks(
        masks_zy,
        orig_spacing_x=0.5,   # 50cm 간격
        target_spacing_x=0.1  # 10cm 간격으로 보간 (원하면 조절)
    )
    np.save(OUT_INTERP, vol_interp)
    print(f"[INFO] Saved INTERP volume: {OUT_INTERP}, shape={vol_interp.shape}")
    print(f"[INFO] num_mid between each pair: {num_mid}")


if __name__ == "__main__":
    main()