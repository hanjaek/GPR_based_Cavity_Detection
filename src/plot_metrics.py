import matplotlib.pyplot as plt

# 성능 값
metrics = ["Dice", "IoU", "PixelAcc"]
values = [0.5608, 0.4555, 0.9814]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values)

plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Lightweight U-Net Segmentation Performance")

# 막대 위에 숫자 표시
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("unet_metrics.png", dpi=300)
plt.show()
