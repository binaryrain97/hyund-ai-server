from flask import Flask, request, jsonify
from preprocess import preprocess_document
from flask_cors import CORS
from cluster_manager import index_document_by_category

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.get_json()
    path = data.get('path')

    if not path:
        return jsonify({'error': '경로가 없습니다.'}), 400

    try:
        result = index_document_by_category(path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
