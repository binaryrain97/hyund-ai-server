import os
import tempfile
from flask import Flask, request, jsonify
from classify import classify_document
from config import init_categories
from preprocess import preprocess_document
from flask_cors import CORS
from search import search_files
from util import allowed_file

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 제한

@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.get_json()
    path = data.get('path')

    if not path:
        return jsonify({'error': '경로가 없습니다.'}), 400

    try:
        result = classify_document(path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/search', methods=['POST'])
def search_file_route():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': '내용이 없습니다.'}), 400

    try:
        result = search_files(query)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/upload_file', methods=['POST'])
def upload_real_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일명이 없습니다.'}), 400

    # 임시 디렉토리에 파일 저장
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = classify_document(tmp_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 임시 파일 삭제 (분류 함수에서 move_file로 이동했다면 삭제 안 해도 됨)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == '__main__':
    init_categories()
    app.run()
