import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GPRCavityDataset
from model import UNet

# ---------------- Dice Loss Function ----------------
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2.0 * (pred * target).sum(dim=(1, 2, 3))
    den = (pred + target).sum(dim=(1, 2, 3)) + eps
    dice = 1.0 - (num / den)
    return dice.mean()

# ---------------- One Training Step ----------------
def train_step(model, loader, optimizer, bce_loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)              # (B,1,H,W) logits
        bce = bce_loss_fn(preds, masks)    # BCEWithLogitsLoss
        dsc = dice_loss(preds, masks)      # Dice loss
        loss = bce + dsc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ---------------- Main Training Loop ----------------
def main():
    # 이 경로만 바뀜: 이제 cavity 전체를 root로 사용
    data_root = "./data/cavity"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # ---------------- Dataset & DataLoader ----------------
    dataset = GPRCavityDataset(root_dir=data_root, transform=None)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # ---------------- Model & Optimizer ----------------
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # ---------------- Training ----------------
    epochs = 50
    os.makedirs("./outputs/checkpoints", exist_ok=True)

    for epoch in range(epochs):
        avg_loss = train_step(model, loader, optimizer, bce_loss_fn, device)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # epoch별 가중치 저장
        ckpt_path = f"./outputs/checkpoints/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)

    print("🎉 Training Complete!")

if __name__ == "__main__":
    main()
