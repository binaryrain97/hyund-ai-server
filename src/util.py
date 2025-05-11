import shutil
import os

def move_file(file_path: str, new_path: str):
    filename = os.path.basename(file_path)
    dest_path = os.path.join(new_path, filename)
    shutil.move(file_path, dest_path)
    return dest_path

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx', 'xls', 'xlsx'}