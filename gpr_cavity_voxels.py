import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1) 스케일 & 경로 설정
# ===========================
SLICE_SPACING_M = 0.5   # 슬라이스 간 간격 (X축)
SCAN_LENGTH_M   = 10.0  # 진행 방향 길이 (Y축)
MAX_DEPTH_M     = 5.0   # 최대 깊이 (Z축)

# 파일 경로: gpr_to_cavity 기준
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_DIR    = os.path.join(BASE_DIR, "test_data")
LABEL_DIR  = os.path.join(BASE_DIR, "ai_hub/src/yolov5_master/runs/detect/exp2/labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# 2) 017_* 이미지 목록 & 크기 파악
# ===========================
img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "017_*.jpg")))
if not img_paths:
    raise RuntimeError(f"017_* 이미지를 찾을 수 없습니다: {IMG_DIR}")

print("[INFO] 사용할 슬라이스:")
for p in img_paths:
    print("  -", os.path.basename(p))

# 슬라이스 개수
num_slices = len(img_paths)

# 하나의 이미지 크기를 라벨에서 유추할 수 없으니,
# test_data 안 017_*.jpg 중 하나의 크기를 가져오려면 cv2를 써야 하는데
# 여기서는 간단히 640x576 (detect.py에서 리사이즈된 크기)라고 가정.
# 더 정확하게 하려면 cv2.imread로 실제 높이/너비 읽어서 쓰면 됨.
H = 576  # 세로(깊이 방향 픽셀)
W = 640  # 가로(진행 방향 픽셀)

# ===========================
# 3) cavity 마스크 볼륨 초기화
# ===========================
# volume_mask[slice, z, y] = 1 이면 그 위치가 cavity
volume_mask = np.zeros((num_slices, H, W), dtype=bool)

for slice_idx, img_path in enumerate(img_paths):
    img_name = os.path.basename(img_path)
    stem, _ = os.path.splitext(img_name)
    label_path = os.path.join(LABEL_DIR, stem + ".txt")

    if not os.path.exists(label_path):
        print(f"[INFO] 라벨 없음, 스킵: {label_path}")
        continue

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # --save-conf 사용: class cx cy w h conf
            if len(parts) == 6:
                cls, cx, cy, w, h, conf = parts
            else:
                cls, cx, cy, w, h = parts
                conf = 1.0

            cls = int(cls)
            cx  = float(cx)
            cy  = float(cy)
            w   = float(w)
            h   = float(h)

            # 여기서는 class 0 만 cavity로 가정
            if cls != 0:
                continue

            # YOLO 정규화 좌표 → 픽셀 좌표
            cx_px = cx * W
            cy_px = cy * H
            w_px  = w * W
            h_px  = h * H

            x1 = int(max(cx_px - w_px / 2, 0))
            x2 = int(min(cx_px + w_px / 2, W - 1))
            y1 = int(max(cy_px - h_px / 2, 0))
            y2 = int(min(cy_px + h_px / 2, H - 1))

            # [y1:y2, x1:x2] 영역을 cavity 로 표시
            volume_mask[slice_idx, y1:y2+1, x1:x2+1] = True

print("[INFO] cavity voxel 개수:", volume_mask.sum())
if volume_mask.sum() == 0:
    raise RuntimeError("cavity 표시된 voxel이 없습니다. 017_* 라벨과 class id를 확인하세요.")

# ===========================
# 4) matplotlib voxels로 시각화
# ===========================
# 좌표 그리드를 m 단위로 만든다.
slice_indices = np.arange(num_slices)     # 0,1,2,...
z_indices     = np.arange(H)
y_indices     = np.arange(W)

X = slice_indices[:, None, None] * SLICE_SPACING_M                    # (S,1,1)
Y = (y_indices[None, None, :] / (W - 1)) * SCAN_LENGTH_M              # (1,1,W)
Z = (z_indices[None, :, None] / (H - 1)) * MAX_DEPTH_M                # (1,H,1)

# matplotlib.voxels는 3D boolean 배열만 필요함.
# 지반 전체 볼륨(박스)에는 얇은 껍데기만 그려보자.
# 여기서는 cavity만 채우고, 겉 박스는 wireframe 처럼 그리기.
cavity = volume_mask

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# cavity를 빨간색으로 표시
# 작은 예시라 downsampling(간격)을 조금 줄여주면 속도가 나음
step = max(1, H // 60)  # 너무 많으면 너무 무거워지니까 샘플링
cav_small = cavity[:, ::step, ::step]
Zs = Z[:, ::step, None]
Ys = Y[:, None, ::step]
Xs = X

ax.voxels(
    Xs, Ys, Zs,
    cav_small,
    facecolors='red',
    edgecolor='k',
    alpha=0.8
)

# 지반 바깥 박스(큐브) 경계선 그리기
max_x = (num_slices - 1) * SLICE_SPACING_M
max_y = SCAN_LENGTH_M
max_z = MAX_DEPTH_M

# 네모선들
for x in [0, max_x]:
    for y0, y1 in [(0, max_y)]:
        ax.plot([x, x], [y0, y0], [0, max_z], color='gray', alpha=0.4)
        ax.plot([x, x], [y1, y1], [0, max_z], color='gray', alpha=0.4)

for y in [0, max_y]:
    for x0, x1 in [(0, max_x)]:
        ax.plot([x0, x1], [y, y], [0, 0], color='gray', alpha=0.4)
        ax.plot([x0, x1], [y, y], [max_z, max_z], color='gray', alpha=0.4)

for z in [0, max_z]:
    ax.plot([0, max_x], [0, 0], [z, z], color='gray', alpha=0.4)
    ax.plot([0, max_x], [max_y, max_y], [z, z], color='gray', alpha=0.4)
    ax.plot([0, 0], [0, max_y], [z, z], color='gray', alpha=0.4)
    ax.plot([max_x, max_x], [0, max_y], [z, z], color='gray', alpha=0.4)

ax.set_xlabel("X (m) - GPR 이동 방향(슬라이스)")
ax.set_ylabel("Y (m) - 진행 방향 (~10 m)")
ax.set_zlabel("Z (m) - 깊이 (~5 m)")
ax.set_title("Site 017 기반 cavity 3D voxel (개념 시각화)")
ax.invert_zaxis()  # 깊이가 아래로 내려가게

out_path = os.path.join(OUTPUT_DIR, "3d_cavity_voxels_017.png")
plt.savefig(out_path, dpi=300)
print("[INFO] 저장:", out_path)

plt.tight_layout()
plt.show()
