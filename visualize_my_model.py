import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 2.2))

# (x위치, 라벨, 타입)
# 타입: enc(파랑), bottleneck(주황), dec(파랑), io(연한 회색)
blocks = [
    (0,  "Input\n(1ch)",        "io"),
    (1,  "inc\n1→16",           "enc"),
    (2,  "down1\n16→32",        "enc"),
    (3,  "down2\n32→64",        "enc"),
    (4,  "down3\n64→128",       "enc"),
    (5,  "down4\n128→128",      "bottleneck"),
    (6,  "up1\n256→64",         "dec"),
    (7,  "up2\n128→32",         "dec"),
    (8,  "up3\n64→16",          "dec"),
    (9,  "up4\n32→16",          "dec"),
    (10, "outc\n16→1",          "io"),
]

h = 1.4    # block height
w = 0.6    # block width
y = 0.3    # block bottom y

colors = {
    "enc": "tab:blue",
    "dec": "tab:blue",
    "bottleneck": "tab:orange",
    "io": "lightgray",
}

# 블럭 그리기
for x, label, btype in blocks:
    rect = Rectangle((x, y), w, h,
                     facecolor=colors[btype],
                     edgecolor="black")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=8)

# 순방향 화살표 (검은색)
for i in range(len(blocks) - 1):
    x_start = blocks[i][0] + w
    x_end   = blocks[i+1][0]
    arrow = FancyArrowPatch(
        (x_start, y + h/2),
        (x_end,   y + h/2),
        arrowstyle="->",
        mutation_scale=10,
        linewidth=1.0,
    )
    ax.add_patch(arrow)

# 스킵 연결 (초록색)
def skip(i_from, i_to, rad):
    x_from = blocks[i_from][0] + w/2
    x_to   = blocks[i_to][0]   + w/2
    arrow = FancyArrowPatch(
        (x_from, y + h),
        (x_to,   y + h),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="->",
        mutation_scale=10,
        linewidth=1.0,
        color="green",
    )
    ax.add_patch(arrow)

# 네 UNet 구조 기준 skip 연결:
# x4 -> up1, x3 -> up2, x2 -> up3, x1 -> up4
skip(4, 6, rad=0.35)  # down3 → up1
skip(3, 7, rad=0.25)  # down2 → up2
skip(2, 8, rad=0.15)  # down1 → up3
skip(1, 9, rad=0.05)  # inc   → up4

ax.set_xlim(-0.5, 10.7)
ax.set_ylim(0, 3)
ax.axis("off")
plt.tight_layout()
plt.show()

# 파일로 저장하고 싶으면:
# plt.savefig("unet_cavity_arch.png", dpi=300, bbox_inches="tight")
