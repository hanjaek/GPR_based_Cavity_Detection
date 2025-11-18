import os
import shutil

# YOLO 탐지 결과 폴더
detect_labels = r"ai_hub/src/yolov5_master/runs/detect/exp3/labels"
# 원본 이미지가 있는 폴더
raw_images_path = r"continuous_data"
# cavity만 복사할 폴더
dst_path = r"classification_cavity_img"

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

    # 원본 이미지 이름 = txt 이름과 동일 + .jpg
    img_name = label_file.replace(".txt", ".jpg")
    raw_img_path = os.path.join(raw_images_path, img_name)

    # 원본 파일이 실제로 존재하면 복사
    if os.path.exists(raw_img_path):
        print(f"[SAVE] Cavity detected → {img_name}")
        shutil.copy(raw_img_path, os.path.join(dst_path, img_name))
    else:
        print(f"[MISS] Raw image not found for {img_name}")
