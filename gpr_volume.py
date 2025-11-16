import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1) 스케일 설정 (가정값)
# ===========================
SLICE_SPACING_M = 0.5   # 슬라이스 간 간격 (X축) 0.5 m
SCAN_LENGTH_M   = 10.0  # 한 이미지의 진행 방향 길이 (Y축) ≒ 10 m 가정
MAX_DEPTH_M     = 5.0   # 한 이미지의 최대 깊이 (Z축) ≒ 5 m 가정

# ===========================
# 2) 경로 설정 (상대 경로)
# ===========================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMG_DIR   = os.path.join(BASE_DIR, "test_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# 3) 017번 site 이미지 불러오기
# ===========================
img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "017_*.jpg")))
if not img_paths:
    raise RuntimeError(f"017_* 이미지를 찾을 수 없습니다: {IMG_DIR}")

print("사용할 이미지:")
for p in img_paths:
    print("  -", os.path.basename(p))

imgs = []
for p in img_paths:
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(f"[WARN] 이미지를 읽을 수 없습니다: {p}")
        continue
    imgs.append(im)

if len(imgs) == 0:
    raise RuntimeError("이미지를 하나도 읽지 못했습니다.")

# 모든 슬라이스가 같은 크기라고 가정
volume = np.stack(imgs, axis=0)  # shape: (num_slices, H, W)
num_slices, H, W = volume.shape
print(f"volume shape = (슬라이스 {num_slices}, 깊이 {H}, 거리 {W})")

# ===========================
# 4) intensity 정규화 & voxel 선택
# ===========================
# 0~1 로 정규화
v_min, v_max = volume.min(), volume.max()
volume_norm = (volume - v_min) / (v_max - v_min + 1e-8)

# 강한 반사만 골라서 voxel로 사용 (threshold는 조절 가능)
THRESH = 0.7
mask = volume_norm > THRESH   # shape: (num_slices, H, W)

voxel_idx = np.argwhere(mask)  # (N, 3)  [slice_idx, z_idx, y_idx]
if len(voxel_idx) == 0:
    raise RuntimeError(f"threshold={THRESH} 기준으로 선택된 voxel이 없습니다. THRESH를 낮춰보세요.")

slice_idx = voxel_idx[:, 0]
z_idx     = voxel_idx[:, 1]
y_idx     = voxel_idx[:, 2]

# ===========================
# 5) 인덱스를 실제 거리/깊이(m)로 변환
# ===========================
# X: 슬라이스 방향
x = slice_idx * SLICE_SPACING_M  # 0, 0.5, 1.0, ...

# Y: 이미지 가로 방향 → 진행 방향 (0 ~ SCAN_LENGTH_M)
y = (y_idx / (W - 1)) * SCAN_LENGTH_M

# Z: 이미지 세로 방향 → 깊이 (0 ~ MAX_DEPTH_M)
z = (z_idx / (H - 1)) * MAX_DEPTH_M

# 색상은 intensity 기반으로
intensity = volume_norm[mask]      # 선택된 voxel의 intensity
colors = plt.cm.viridis(intensity) # colormap 적용

print("선택된 voxel 수:", len(x))

# ===========================
# 6) 3D 시각화
# ===========================
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, s=5, c=colors, alpha=0.8)

ax.set_xlabel("X (m) - 슬라이스 방향 (017_1, 017_2, ...)")
ax.set_ylabel("Y (m) - 진행 방향 (~10 m)")
ax.set_zlabel("Z (m) - 깊이 (~5 m)")
ax.set_title("Site 017 GPR intensity 기반 3D voxel point cloud")

ax.invert_zaxis()  # 깊이 방향은 아래로 내려가게

output_path = os.path.join(OUTPUT_DIR, "3d_volume_017.png")
plt.savefig(output_path, dpi=300)
print(f"3D 볼륨 시각화 이미지 저장 완료: {output_path}")

plt.tight_layout()
plt.show()
