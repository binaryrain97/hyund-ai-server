import shutil
import os

def move_file_to_category(file_path: str, category_kor: str):
    from config import BASE_PATH, CATEGORY_KOR_TO_ENG

    # 영문 폴더명으로 매핑
    category_eng = CATEGORY_KOR_TO_ENG.get(category_kor, "uncategorized")
    
    # 대상 폴더 경로 생성
    dest_dir = os.path.join(BASE_PATH, category_eng)
    os.makedirs(dest_dir, exist_ok=True)

    # 대상 경로
    filename = os.path.basename(file_path)
    dest_path = os.path.join(dest_dir, filename)

    # 파일 이동
    shutil.move(file_path, dest_path)
    print(f"Moved '{file_path}' to '{dest_path}'")

    return dest_path
