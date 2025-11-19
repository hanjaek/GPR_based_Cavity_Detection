import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 2.3))

blocks = [
    (0,  "GPR\n원시데이터",         "data"),
    (1.5,"XY/YZ/XZ\n단면 이미지",   "data"),
    (3.0,"바운딩박스\n라벨링(JSON)", "proc"),
    (4.5,"Train / Valid /\nTest 분할", "proc"),
    (6.0,"YOLOv5\n학습",            "model"),
    (7.5,"탐지 모델\n(mAP 85.76%)", "model_out"),
]

h, w, y = 1.4, 1.0, 0.3

colors = {
    "data": "lightblue",
    "proc": "lightgreen",
    "model": "orange",
    "model_out": "gold",
}

# 블럭 그리기
for x, label, btype in blocks:
    rect = Rectangle((x, y), w, h,
                     facecolor=colors[btype],
                     edgecolor="black")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=9)

# 좌→우 화살표
for i in range(len(blocks) - 1):
    x_start = blocks[i][0] + w
    x_end   = blocks[i+1][0]
    arrow = FancyArrowPatch(
        (x_start, y + h/2),
        (x_end,   y + h/2),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.0,
    )
    ax.add_patch(arrow)

ax.set_xlim(-0.5, 9)
ax.set_ylim(0, 2.5)
ax.axis("off")
plt.tight_layout()
plt.show()

# 저장하고 싶으면:
# plt.savefig("aihub_yolov5_pipeline.png", dpi=300, bbox_inches="tight")
