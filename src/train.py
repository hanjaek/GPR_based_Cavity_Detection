# ==================================================
# Train UNet for Cavity Segmentation
# ==================================================

import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import GPRCavityDataset
from model import UNet


# ==================================================
# Main Training Loop
# ==================================================
def main():
    # --------------------------------------------------
    # 경로 설정 (src 기준)
    # --------------------------------------------------
    img_root = "../data/cavity"
    mask_root = "../data_mask/cavity"

    batch_size = 4
    lr = 1e-3
    num_epochs = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # --------------------------------------------------
    # Dataset & Loader
    # --------------------------------------------------
    dataset = GPRCavityDataset(img_root, mask_root)

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # --------------------------------------------------
    # 모델 / 손실함수 / Optimizer
    # --------------------------------------------------
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = 9999.0

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # ============================
        # Train Step
        # ============================
        model.train()
        train_loss_sum = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / n_train

        # ============================
        # Validation Step
        # ============================
        model.eval()
        val_loss_sum = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)

        val_loss = val_loss_sum / n_val

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