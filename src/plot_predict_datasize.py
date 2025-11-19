import matplotlib.pyplot as plt
import numpy as np

# 실제/예상 데이터 포인트
data_sizes = np.array([76, 300, 500, 1000])

# Dice: 실제(0.5608) + 예측치
dice_scores = np.array([0.5608, 0.65, 0.72, 0.82])

# IoU: 실제(0.4555) + 예측치
iou_scores  = np.array([0.4555, 0.55, 0.60, 0.70])

plt.figure(figsize=(7, 4))

# -------------------------------
# 그라데이션 컬러맵
# -------------------------------
dice_colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(data_sizes)))    # 파랑 그라데이션
iou_colors  = plt.cm.Greens(np.linspace(0.4, 1.0, len(data_sizes)))   # 초록 그라데이션

# -------------------------------
# Dice 라인 + 점 (실선)
# -------------------------------
plt.plot(data_sizes, dice_scores, color=dice_colors[-1], linewidth=2.3, label="Dice (Projected)")
for i in range(len(data_sizes)):
    plt.scatter(data_sizes[i], dice_scores[i], color=dice_colors[i], s=90, zorder=5)

# -------------------------------
# IoU 라인 + 점 (실선)
# -------------------------------
plt.plot(data_sizes, iou_scores, color=iou_colors[-1], linewidth=2.3, label="IoU (Projected)")
for i in range(len(data_sizes)):
    plt.scatter(data_sizes[i], iou_scores[i], color=iou_colors[i], s=90, zorder=5)

# -------------------------------
# 텍스트 라벨
# -------------------------------
# plt.text(76, dice_scores[0] + 0.02, "Current Dice (0.56)", ha="center", va="bottom")
# plt.text(76, iou_scores[0]  - 0.04, "Current IoU (0.46)", ha="center", va="bottom")

# 축/제목 (영문)
plt.xlabel("Number of Training Samples")
plt.ylabel("Score")
plt.title("Projected Dice and IoU Improvement with Additional Data")

# 축 범위
plt.ylim(0.4, 0.9)
plt.xlim(0, 1050)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()
