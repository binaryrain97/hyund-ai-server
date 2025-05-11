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
    
# @app.route('/upload-file', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': '파일이 없습니다'}), 400
        
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': '선택된 파일이 없습니다'}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({'error': '허용되지 않은 파일 형식'}), 400

    # try:
    #     # 임시 저장
    #     filename = secure_filename(file.filename)
    #     temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(temp_path)

    #     # 문서 처리
    #     processed_data = preprocess_document(temp_path)
    #     category_result = index_document_by_category(processed_data)

    #     # 최종 이동
    #     final_path = os.path.join(app.config['FINAL_FOLDER'], filename)
    #     os.rename(temp_path, final_path)

    #     return jsonify({
    #         'filename': filename,
    #         'category': category_result['category'],
    #         'saved_path': final_path,
    #         'status': 'success'
    #     }), 200

    # except Exception as e:
    #     # 오류 발생 시 임시 파일 정리
    #     if os.path.exists(temp_path):
    #         os.remove(temp_path)
    #     return jsonify({'error': str(e)}), 500

    # finally:
    #     # 임시 파일 존재 시 항상 삭제
    #     if 'temp_path' in locals() and os.path.exists(temp_path):
    #         os.remove(temp_path)

if __name__ == '__main__':
    init_categories()
    app.run()
