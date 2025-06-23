from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from app.inference import predict

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_audio():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file received'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    label, probs = predict(path)
    return jsonify({'prediction': label, 'confidence': probs})

if __name__ == '__main__':
    print("🚀 SonicShield is starting...")
    app.run(debug=True)