import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ===========================
# 1) 스케일 설정 (가정값)
# ===========================
SLICE_SPACING_M = 0.5   # 슬라이스 간 간격 (X축) 0.5 m
SCAN_LENGTH_M  = 10.0   # 한 이미지의 진행 방향 길이 (Y축) ≒ 10 m 가정
MAX_DEPTH_M    = 5.0    # 한 이미지의 최대 깊이 (Z축) ≒ 5 m 가정

# ===========================
# 2) 경로 설정
# ===========================
IMG_DIR   = r"C:\Users\hjk25\gpr_to_cavity\test_data"
LABEL_DIR = r"C:\Users\hjk25\gpr_to_cavity\ai_hub\src\yolov5_master\runs\detect\exp\labels"

# ===========================
# 3) 001번 site만 선택
# ===========================
# 001_1.jpg, 001_2.jpg, ... 형태만 읽기
image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "001_*.jpg")))

if not image_paths:
    raise RuntimeError(f"001_* 이미지를 찾을 수 없습니다: {IMG_DIR}")

print("사용할 이미지:")
for p in image_paths:
    print("  -", os.path.basename(p))

points = []        # (x, y, z)
point_labels = []  # class id

for slice_idx, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    stem, _ = os.path.splitext(img_name)

    # 이 이미지에 해당하는 YOLO 라벨 파일 경로
    label_path = os.path.join(LABEL_DIR, stem + ".txt")

    if not os.path.exists(label_path):
        print(f"[INFO] 라벨 없음, 스킵: {label_path}")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] 이미지를 읽을 수 없습니다: {img_path}")
        continue

    H, W = img.shape[:2]

    # 이 슬라이스의 X 좌표 = 슬라이스 인덱스 × 0.5m
    x_coord = slice_idx * SLICE_SPACING_M

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # --save-conf 쓴 경우: class cx cy w h conf
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

            # YOLO 정규화 → 픽셀
            cx_px = cx * W
            cy_px = cy * H
            w_px  = w * W
            h_px  = h * H

            # 박스 안에서 3x3 포인트 샘플링
            nx, ny = 3, 3
            xs = np.linspace(cx_px - w_px / 2, cx_px + w_px / 2, nx)
            ys = np.linspace(cy_px - h_px / 2, cy_px + h_px / 2, ny)

            for px in xs:
                for py in ys:
                    # 픽셀 → 실제 좌표 (y, z)
                    y_coord = (px / W) * SCAN_LENGTH_M   # 진행 방향
                    z_coord = (py / H) * MAX_DEPTH_M     # 깊이

                    points.append([x_coord, y_coord, z_coord])
                    point_labels.append(cls)

points = np.array(points)
point_labels = np.array(point_labels)

print("총 포인트 수:", len(points))
if len(points) == 0:
    raise RuntimeError("포인트가 0개입니다. 001_*.txt에 탐지 결과가 있는지 확인하세요.")

# ===========================
# 4) 3D 시각화
# ===========================
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

# class에 따라 색상 변경 (0=cavity, 1=box 라고 가정)
colors = np.where(point_labels == 0, 'red', 'blue')

ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           s=20, c=colors, alpha=0.8)

ax.set_xlabel("X (m) - 슬라이스 방향 (001_1, 001_2, ...)")
ax.set_ylabel("Y (m) - 진행 방향 (~10 m)")
ax.set_zlabel("Z (m) - 깊이 (~5 m)")
ax.set_title("Site 001 GPR YOLO 3D 포인트 클라우드")

plt.tight_layout()
plt.show()
