# ==================================================
# Inference Script for Trained UNet
# - GPR 이미지 입력 → cavity mask 자동 생성
# - 결과는 src 밖의 prediction_img 폴더에 저장
# ==================================================

import os
import argparse
import numpy as np
from PIL import Image
import torch

from model import UNet


# ==================================================
# 이미지 전처리 함수
# ==================================================
def load_image_as_tensor(img_path, device):
    """
    ==================================================
    입력 이미지 로딩 & 전처리
    - grayscale 로 변환
    - 0~1 정규화
    - (1,1,H,W) 텐서로 변환
    ==================================================
    """
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, img_np


# ==================================================
# 저장 헬퍼 함수
# ==================================================
def save_mask(mask_np, save_path):
    """
    ==================================================
    sigmoid 확률 → 0/255 binary 이미지 저장
    ==================================================
    """
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    mask_img.save(save_path)
    print(f"[INFO] Saved mask: {save_path}")


# ==================================================
# MAIN
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input GPR image path"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output mask filename (optional)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # --------------------------------------------------
    # 폴더 생성: src 밖 prediction_img/
    # --------------------------------------------------
    save_dir = "../prediction_img"
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # 모델 로드
    # --------------------------------------------------
    checkpoint_path = "../checkpoints/unet_best.pth"
    model = UNet(n_channels=1, n_classes=1).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # --------------------------------------------------
    # 이미지 전처리
    # --------------------------------------------------
    input_path = args.input
    img_tensor, orig_np = load_image_as_tensor(input_path, device)

    # --------------------------------------------------
    # 추론
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # --------------------------------------------------
    # 후처리 → Binary Mask
    # --------------------------------------------------
    mask_np = (prob > 0.5).astype(np.uint8) * 255

    # --------------------------------------------------
    # 저장 경로 결정
    # --------------------------------------------------
    if args.output_name is None:
        base = os.path.basename(input_path)
        fname = os.path.splitext(base)[0] + "_mask.png"
    else:
        fname = args.output_name

    save_path = os.path.join(save_dir, fname)

    # --------------------------------------------------
    # 저장
    # --------------------------------------------------
    save_mask(mask_np, save_path)


# ==================================================
if __name__ == "__main__":
    main()
# ==================================================
