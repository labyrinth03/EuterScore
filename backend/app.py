from flask import Flask, request, jsonify
import os
import librosa
from magenta.models.onsets_frames import infer
from magenta.models.onsets_frames import sequence_proto_to_midi_file
from google.protobuf import text_format
import tensorflow as tf

app = Flask(__name__)

# 사전 훈련된 Onsets and Frames 모델 로드
model = infer.OnsetsAndFrames()

# 업로드된 파일 저장 경로
UPLOAD_FOLDER = './static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 오디오 파일을 MIDI로 변환하는 함수
def convert_audio_to_midi(audio_file_path):
    # 오디오 파일 로드
    sequence = model.infer(audio_file_path)
    
    # MIDI 파일로 변환
    midi_path = os.path.join(UPLOAD_FOLDER, 'output.midi')
    sequence_proto_to_midi_file(sequence, midi_path)
    return midi_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 오디오 파일 저장
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # 오디오 파일을 MIDI로 변환
        midi_file_path = convert_audio_to_midi(file_path)
        
        return jsonify({'message': 'File successfully processed', 'midi_file': midi_file_path}), 200

if __name__ == "__main__":
    app.run(debug=True)
