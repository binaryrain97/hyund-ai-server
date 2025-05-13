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

    # 원본 파일명 추출
    original_name = file.filename
    temp_dir = tempfile.TemporaryDirectory()  # 임시 디렉토리 생성

    try:
        # 1. 원본 파일명으로 임시 저장
        temp_path = os.path.join(temp_dir.name, original_name)
        file.save(temp_path)

        # 2. 파일 분류 처리
        result = classify_document(temp_path)
        
        # 3. 처리 완료 후 임시 디렉토리 정리 (파일은 이미 이동됨)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 4. 임시 디렉토리 삭제 (내부 파일은 이미 이동 완료)
        try:
            temp_dir.cleanup()
        except Exception:
            pass

if __name__ == '__main__':
    init_categories()
    app.run(debug=True)
