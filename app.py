from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import torch

from app.model import AudioTransformer
from app.preprocessing import preprocess_audio

# Define class names (update as per your training classes)
class_names = ['gunshot', 'scream', 'glass_break', 'background_noise']

# Set up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device("cpu")
model = AudioTransformer(num_classes=len(class_names))
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.to(device)
model.eval()

# Home route (optional — if you're using index.html)
@app.route('/')
def home():
    return "<h1>SonicShield Backend is Live 🛡️</h1>"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_audio():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file received'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        audio_tensor = preprocess_audio(path)
        audio_tensor = audio_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(audio_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

        prediction = class_names[top_idx.item()]
        confidence = float(top_prob)

        return jsonify({'prediction': prediction, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 SonicShield is starting...")
    app.run(debug=True)
