# ==================================================
# Train UNet for Cavity Segmentation
# - BCE + Dice Loss 조합
# - 데이터 증강 미적용
# - LR Scheduler 적용
# ==================================================

import os

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import GPRCavityDataset
from model import UNet

# ============================
# Segmentation Metrics
# ============================
def compute_metrics(logits, masks, threshold: float = 0.5):
    """
    logits, masks: (B,1,H,W) tensor
    return: (dice, iou, pixel_acc)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    masks_flat = masks.view(masks.size(0), -1)

    intersection = (preds_flat * masks_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - intersection

    eps = 1e-7
    dice = (2 * intersection + eps) / (preds_flat.sum(dim=1) + masks_flat.sum(dim=1) + eps)
    iou = (intersection + eps) / (union + eps)
    acc = (preds_flat == masks_flat).float().mean(dim=1)

    return dice.mean().item(), iou.mean().item(), acc.mean().item()


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
# Main Training Loop (BCE + Dice, Aug OFF)
# ==================================================
def main():
    # --------------------------------------------------
    # 경로 설정 (src 기준)
    # --------------------------------------------------
    img_root = "../data2"
    mask_root = "../data2_mask"

    batch_size = 4
    lr = 1e-3
    num_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Mode  : BCE + Dice, Augment OFF")

    # --------------------------------------------------
    # Dataset & Loader (증강 없음)
    # --------------------------------------------------
    dataset = GPRCavityDataset(
        img_root=img_root,
        mask_root=mask_root,
        transform=None  # 증강 미적용
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

    def loss_fn(logits, masks):
        # BCE + Dice 조합
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
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / n_train

        # ============================
        # Validation Step
        # ============================
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum  = 0.0
        val_acc_sum  = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)

                loss = loss_fn(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)

                d, i, a = compute_metrics(logits, masks, threshold=0.5)
                val_dice_sum += d * imgs.size(0)
                val_iou_sum  += i * imgs.size(0)
                val_acc_sum  += a * imgs.size(0)

        val_loss = val_loss_sum / n_val
        val_dice = val_dice_sum / n_val
        val_iou  = val_iou_sum  / n_val
        val_acc  = val_acc_sum  / n_val

        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"Train={train_loss:.4f}  "
            f"Val={val_loss:.4f}  "
            f"Dice={val_dice:.4f}  "
            f"IoU={val_iou:.4f}  "
            f"PixAcc={val_acc:.4f}"
        )

        # ============================
        # Best Model Save
        # ============================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("../checkpoints", exist_ok=True)
            save_path = "../checkpoints/train_bce_dice_noaug.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   -> Best model saved: {save_path}")


if __name__ == "__main__":
    main()
