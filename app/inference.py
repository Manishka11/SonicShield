import torch
import librosa
import numpy as np
from app.model import AudioTransformer # Make sure this matches your model class path

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels (update these if different)
CLASS_LABELS = [
    'gun_shot',
    'scream',
    'glass_breaking',
    'siren',
    'engine_idling',
    'normal'
]

# Load trained model
model = AudioTransformer(num_classes=10)
model.load_state_dict(torch.load("model/model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocess audio into mel spectrogram tensor
def preprocess_audio(path, sample_rate=16000, n_mels=128):
    y, sr = librosa.load(path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)  # Normalize
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, H, W]
    return mel_tensor

# Make prediction
def predict(audio_path):
    input_tensor = preprocess_audio(audio_path).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
        predicted_index = np.argmax(probabilities)
        predicted_label = CLASS_LABELS[predicted_index]
        return predicted_label, [float(p) for p in probabilities]
