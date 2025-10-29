import os
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from model import UNet

# ---------------- 단일 이미지 로딩 ----------------
def load_image_as_tensor(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    img_t = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0  # [3,H,W]
    return img_t.unsqueeze(0)  # [1,3,H,W] 배치 차원 추가

# ---------------- 예측 및 저장 ----------------
def run_inference_on_cavity(model, cavity_root, save_root, device):
    os.makedirs(save_root, exist_ok=True)

    # cavity_root 예: ./data/cavity
    for site_name in sorted(os.listdir(cavity_root)):
        site_path = os.path.join(cavity_root, site_name)
        img_dir = os.path.join(site_path, "images")

        if not os.path.isdir(img_dir):
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            img_path = os.path.join(img_dir, fname)

            # 이미지 로드
            image = load_image_as_tensor(img_path).to(device)

            # 예측
            model.eval()
            with torch.no_grad():
                pred_logits = model(image)          # [1,1,H,W] (raw logits)
                pred_prob   = torch.sigmoid(pred_logits)
                pred_mask   = (pred_prob > 0.5).float()  # binary mask

            # 저장 파일명 예:
            # site_001_001_1_pred.png
            base = os.path.splitext(fname)[0]  # 001_1
            save_name = f"{site_name}_{base}_pred.png"
            save_path = os.path.join(save_root, save_name)

            save_image(pred_mask, save_path)
            print(f"✅ Saved: {save_path}")

# ---------------- main ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Using device: {device}")

    # 모델 준비
    model = UNet(in_channels=3, out_channels=1).to(device)

    checkpoint_path = "./outputs/checkpoints/epoch_50.pth"  # 필요시 바꿔
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"📦 Loaded checkpoint: {checkpoint_path}")

    cavity_root = "./data/cavity"
    save_root   = "./outputs/predictions"

    run_inference_on_cavity(model, cavity_root, save_root, device)

    print("🎉 Inference complete! Check outputs/predictions/")

if __name__ == "__main__":
    main()
