import shutil
from pathlib import Path

# 현재 generate_img.py 는 test 폴더 안에 있으므로
# 원본 이미지는 test/test1/1.png 경로에 있음
folder = Path(__file__).resolve().parent / "test1"

src = folder / "1.png"

start = 2
end = 1000

for i in range(start, end + 1):
    dst = folder / f"{i}.png"
    shutil.copy(src, dst)

print("완료: 2.png ~ 1000.png 생성")