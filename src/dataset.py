# ==================================================
# GPR Cavity Dataset Loader
# - 원본 GPR 이미지 + cavity mask 매칭
# - (site_xxx / filename) 구조를 자동 정렬
# ==================================================

import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


# ==================================================
# GPRCavityDataset
# ==================================================
class GPRCavityDataset(Dataset):
    """
    ==================================================
    GPR + Mask 쌍을 로딩하는 Dataset 클래스
    - 이미지:  data/cavity/site_xxx/xxx.jpg
    - 마스크:  data_mask/cavity/site_xxx/xxx.jpg
    ==================================================
    """

    def __init__(self, img_root, mask_root, transform=None):
        self.samples = []
        self.transform = transform

        # --------------------------------------------------
        # 폴더 자동 탐색
        # --------------------------------------------------
        site_dirs = sorted(glob.glob(os.path.join(img_root, "site_*")))

        for site_dir in site_dirs:
            site_name = os.path.basename(site_dir)
            mask_site_dir = os.path.join(mask_root, site_name)

            img_paths = sorted(glob.glob(os.path.join(site_dir, "*.jpg")))

            for img_path in img_paths:
                fname = os.path.basename(img_path)
                mask_path = os.path.join(mask_site_dir, fname)

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
                else:
                    print(f"[WARN] mask not found for {img_path}")

        print(f"[INFO] Total matched pairs: {len(self.samples)}")

    # ------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # ==============================
        # 이미지 로딩
        # ==============================
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)

        # cavity 영역 이진화
        mask = (mask > 10).astype(np.float32)

        # (H,W) -> (1,H,W)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask