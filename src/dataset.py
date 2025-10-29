import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class GPRCavityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir 예:
            ./data/cavity

        내부 구조 예:
            data/
              cavity/
                site_001/
                  images/
                    001_1.jpg
                    001_2.jpg
                  masks/
                    001_1_mask.png
                    001_2_mask.png
                site_002/
                  images/
                  masks/
                ...

        이 Dataset은 cavity 안의 모든 site_* 폴더를 순회하면서
        (image_path, mask_path) 쌍을 수집한다.
        """
        self.samples = []
        self.transform = transform

        # ---------------- 모든 site_* 폴더 순회 ----------------
        for site_name in sorted(os.listdir(root_dir)):
            site_path = os.path.join(root_dir, site_name)
            if not os.path.isdir(site_path):
                continue

            img_dir = os.path.join(site_path, "images")
            mask_dir = os.path.join(site_path, "masks")

            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                continue

            # 해당 site 안의 이미지들 다 탐색
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                base = os.path.splitext(fname)[0]          # 예: "001_1"
                img_path = os.path.join(img_dir, fname)    # 예: site_001/images/001_1.jpg
                mask_path = os.path.join(
                    mask_dir,
                    base + "_mask.png"                     # 예: site_001/masks/001_1_mask.png
                )

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
                else:
                    # 마스크 없는 경우는 그냥 건너뛰어 (학습 불가니까)
                    pass

        # 이제 self.samples = [(img_path, mask_path), ...]
        # 예: ("data/cavity/site_001/images/001_1.jpg", "data/cavity/site_001/masks/001_1_mask.png")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # ---------------- 이미지 & 마스크 로드 ----------------
        image = Image.open(img_path).convert("RGB")   # (H,W,3)
        mask = Image.open(mask_path).convert("L")     # (H,W) 0~255

        image_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        # ---------------- 마스크 이진화 (0 또는 1) ----------------
        mask_np = (mask_np > 0).astype(np.float32)

        # ---------------- transform 적용 (증강 등) ----------------
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented["image"]
            mask_np = augmented["mask"]

        # ---------------- Tensor 변환 ----------------
        image_t = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0   # [3,H,W]
        mask_t = torch.tensor(mask_np).unsqueeze(0).float()                 # [1,H,W]

        return image_t, mask_t
