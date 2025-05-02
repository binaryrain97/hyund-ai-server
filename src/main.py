from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    return jsonify({
        'result' : True
        })

if __name__ == '__main__':
    app.run()
