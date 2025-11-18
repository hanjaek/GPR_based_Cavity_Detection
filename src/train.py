# ==================================================
# Train UNet for Cavity Segmentation (Improved Ver.)
# - BCE + Dice Loss 조합
# - 간단한 데이터 증강 포함
# - LR Scheduler 적용
# ==================================================

import os
import random

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import GPRCavityDataset
from model import UNet


# ==================================================
# Dice Loss 정의
# ==================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B,1,H,W), targets: (B,1,H,W)
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


# ==================================================
# 간단한 데이터 증강 함수
#  - 좌우 반전
#  - 밝기 조정 (0.9 ~ 1.1배)
# ==================================================
def simple_augment(img, mask):
    """
    img, mask: (1,H,W) tensor
    """
    # 좌우 반전
    if random.random() < 0.5:
        img = torch.flip(img, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # 밝기 조정
    if random.random() < 0.5:
        factor = 0.9 + 0.2 * random.random()  # 0.9 ~ 1.1
        img = img * factor
        img = torch.clamp(img, 0.0, 1.0)

    return img, mask


# ==================================================
# Main Training Loop
# ==================================================
def main():
    # --------------------------------------------------
    # 경로 설정 (src 기준)
    # --------------------------------------------------
    img_root = "../data2"
    mask_root = "../data2_mask"

    batch_size = 4
    lr = 1e-3
    num_epochs = 50  # 기존 30 → 50으로 살짝 증가

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # --------------------------------------------------
    # Dataset & Loader
    # --------------------------------------------------
    dataset = GPRCavityDataset(
        img_root=img_root,
        mask_root=mask_root,
        transform=simple_augment  # 증강 적용
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # --------------------------------------------------
    # 모델 / 손실함수 / Optimizer / Scheduler
    # --------------------------------------------------
    model = UNet(n_channels=1, n_classes=1).to(device)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    def combined_loss(logits, masks):
        return 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_loss = 9999.0

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # ============================
        # Train Step
        # ============================
        model.train()
        train_loss_sum = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / n_train

        # ============================
        # Validation Step
        # ============================
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = combined_loss(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)

        val_loss = val_loss_sum / n_val
        scheduler.step()

        print(f"[Epoch {epoch:03d}] Train={train_loss:.4f}  Val={val_loss:.4f}")

        # ============================
        # Best Model Save
        # ============================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("../checkpoints", exist_ok=True)
            save_path = "../checkpoints/unet_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   -> Best model saved: {save_path}")


if __name__ == "__main__":
    main()
