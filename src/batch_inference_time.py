# ==================================================
# Batch Inference for Cavity Segmentation + Time Measure
# (continuous_img 폴더 기준)
# ==================================================

import os
import glob
import numpy as np
from PIL import Image
import torch
import time  # ★ 시간 측정

from model import UNet


# ---------------- 이미지 로딩 ----------------
def load_image_as_tensor(img_path, device):
    """
    입력 이미지 로딩 & 전처리
    - grayscale 변환
    - 0~1 정규화
    - (1,1,H,W) 텐서로 변환
    """
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def save_mask(mask_np, save_path):
    """0/255 binary mask를 PNG로 저장"""
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    mask_img.save(save_path)


def main():
    # --------------------------------------------------
    # 경로 설정
    # --------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    img_dir = os.path.join(base_dir, "continuous_data")      # ★ 변경됨
    out_dir = os.path.join(base_dir, "continuous_mask_time")  # ★ 저장 폴더

    ckpt_path = os.path.join(base_dir, "checkpoints", "unet_best.pth")

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # --------------------------------------------------
    # 모델 로드
    # --------------------------------------------------
    model = UNet(n_channels=1, n_classes=1).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # --------------------------------------------------
    # continuous_img에서 이미지 목록 로드
    # --------------------------------------------------
    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths = sorted(img_paths)

    print(f"[INFO] Found {len(img_paths)} images in {img_dir}")

    # --------------------------------------------------
    # ★★★ 시간 측정 시작 ★★★
    # --------------------------------------------------
    start = time.perf_counter()

    # --------------------------------------------------
    # 추론 시작
    # --------------------------------------------------
    for idx, img_path in enumerate(img_paths, 1):
        base_name = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(base_name)[0]
        save_path = os.path.join(out_dir, name_wo_ext + "_mask.png")

        tensor = load_image_as_tensor(img_path, device)

        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask_np = (prob > 0.3).astype(np.uint8) * 255

        save_mask(mask_np, save_path)

    # --------------------------------------------------
    # ★★★ 시간 측정 종료 ★★★
    # --------------------------------------------------
    end = time.perf_counter()

    total_time = end - start
    per_image = total_time / len(img_paths)

    print("\n===================== RESULT =====================")
    print(f"[RESULT] Total Inference Time: {total_time:.2f} sec")
    print(f"[RESULT] Per Image: {per_image:.5f} sec/image")
    print("==================================================")
    print("[INFO] Batch inference finished.")


if __name__ == "__main__":
    main()