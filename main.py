from flask import Flask, request, jsonify
from flask_cors import CORS
from backend import PrescriptionBackend

app = Flask(__name__)
CORS(app)
backend = PrescriptionBackend()

@app.route('/transcribe_stream', methods=['POST'])
def transcribe_stream():
    audio_file = request.files.get('audio')
    result, status_code = backend.process_transcription_request(audio_file)
    return jsonify(result), status_code

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('text')
    result, status_code = backend.process_chat_request(user_input)
    return jsonify(result), status_code

@app.route('/save_prescription', methods=['POST'])
def save_prescription():
    data = request.json
    result, status_code = backend.save_prescription_data(data)
    return jsonify(result), status_code

if __name__ == '__main__':
    app.run(debug=True)
