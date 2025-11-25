import numpy as np
from pathlib import Path
from PIL import Image
import re

# ================================================================
# 설정
# ================================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# 마스크 이미지들이 들어 있는 폴더
# (필요하면 여기 경로만 바꿔서 재사용하면 됨)
MASK_DIR = PROJECT_ROOT / "test" / "test3"

# 지원하는 이미지 확장자
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# 출력 볼륨 파일 경로
OUT_PATH = THIS_DIR / "cavity_volume.npy"

# (정보 출력용) 실제 GPR 슬라이스 간격 (m 단위, 필요시 수정)
SLICE_SPACING_X = 0.5  # 예: 0.5m (50cm)


def natural_key(path: Path):
    """
    파일명을 숫자 기준으로 정렬하기 위한 key.
    예: cavity_001.png, cavity_002.png, cavity_010.png 순서를 맞춰줌.
    """
    s = path.stem
    nums = re.findall(r"\d+", s)
    return int(nums[0]) if nums else s


def load_mask_slices(mask_dir: Path) -> np.ndarray:
    """
    마스크 이미지(YZ 슬라이스)들을 읽어 X 방향으로 쌓아
    하나의 3D 볼륨 (z, y, x)을 생성한다.

    - 각 이미지는 YZ 평면이라고 가정 (세로: z, 가로: y)
    - 파일명 숫자 기준으로 정렬하여 X 축 순서를 만든다.
    """
    img_paths = [
        p for p in mask_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not img_paths:
        raise FileNotFoundError(f"No image files found in {mask_dir}")

    # cavity_yz_MALA_000001_mask.png 처럼 숫자 포함된 이름도 잘 정렬되도록 처리
    img_paths = sorted(img_paths, key=natural_key)

    masks = []
    for p in img_paths:
        img = Image.open(p).convert("L")   # Grayscale
        arr = np.array(img, dtype=np.float32)

        # 흰색(>0)을 공동(1), 나머지를 지반(0)으로 처리
        mask = (arr > 0).astype(np.uint8)  # (z, y)

        masks.append(mask)

    vol = np.stack(masks, axis=-1)  # (z, y, x)

    print(f"[INFO] Loaded {len(img_paths)} slices from: {mask_dir}")
    print(f"[INFO] Volume shape (z, y, x): {vol.shape}")
    print(f"[INFO] Approx. X length ≈ {vol.shape[-1] * SLICE_SPACING_X:.2f} m")

    return vol


def main():
    vol = load_mask_slices(MASK_DIR)
    np.save(OUT_PATH, vol)
    print(f"[INFO] Saved volume to: {OUT_PATH}")


if __name__ == "__main__":
    main()