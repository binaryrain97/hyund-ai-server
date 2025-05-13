import shutil
import os

def move_file(src_path: str, dest_dir: str):
    # 디렉토리 생성
    os.makedirs(dest_dir, exist_ok=True)
    
    # 원본 파일명 추출
    file_name = os.path.basename(src_path)
    
    # 최종 경로
    dest_path = os.path.join(dest_dir, file_name)
    
    # 파일 이동 (크로스 디바이스 대응)
    shutil.move(src_path, dest_path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx', 'xls', 'xlsx'}