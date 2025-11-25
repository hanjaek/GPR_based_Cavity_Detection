import shutil
from pathlib import Path

# test1 폴더 경로
folder = Path("test/test1")

# 원본 파일
src = folder / "1.png"

# 복사 생성할 파일 개수
start = 2
end = 1000

for i in range(start, end + 1):
    dst = folder / f"{i}.png"
    shutil.copy(src, dst)

print("- 2.png ~ 1000.png 생성 완료")