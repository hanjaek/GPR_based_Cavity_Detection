import os
import shutil

# 현재 파일(src 내부) 기준 프로젝트 루트 경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# YOLO 탐지 결과 라벨 폴더
detect_labels = os.path.join(
    BASE_DIR,
    "ai_hub",
    "src",
    "yolov5_master",
    "runs",
    "detect",
    "exp3",
    "labels"
)

# 원본 연속 이미지 폴더
raw_images_path = os.path.join(BASE_DIR, "continuous_data")

# cavity만 복사할 폴더
dst_path = os.path.join(BASE_DIR, "classification_cavity_img")
os.makedirs(dst_path, exist_ok=True)

# cavity class ID
CAVITY_ID = 1

for label_file in os.listdir(detect_labels):
    if not label_file.endswith(".txt"):
        continue

    txt_path = os.path.join(detect_labels, label_file)

    # txt 내부 확인
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # cavity가 탐지된 파일인지 확인
    has_cavity = any(line.startswith(str(CAVITY_ID)) for line in lines)
    if not has_cavity:
        continue

    # 이미지 파일명
    img_name = label_file.replace(".txt", ".jpg")
    raw_img_path = os.path.join(raw_images_path, img_name)

    # 존재하면 복사
    if os.path.exists(raw_img_path):
        print(f"[SAVE] Cavity detected → {img_name}")
        shutil.copy(raw_img_path, os.path.join(dst_path, img_name))
    else:
        print(f"[MISS] Raw image not found for {img_name}")