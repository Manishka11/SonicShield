# SonicShield: Real-Time Audio Threat Intelligence

SonicShield is an AI-driven system designed to analyze and classify environmental audio in real time, flagging potential threat signatures such as gunshots, sirens, and machinery. Built using deep learning on the ESC-50 and UrbanSound8K datasets, it empowers intelligent surveillance and smart safety solutions.

## Live Demo
Try the app now on [Hugging Face Spaces](https://huggingface.co/spaces/mishkicodes/manishka)

---

## Project Structure
```
SonicShield/
├── app.py                # Gradio frontend + backend
├── requirements.txt      # Python dependencies
├── model.pth             # Trained PyTorch model
├── app/
│   ├── model.py          # AudioTransformer model class
│   └── preprocessing.py  # Audio preprocessing pipeline
```

---

## Features
- Upload or record an audio clip
- Real-time prediction of sound class
- Displays confidence scores
- Powered by a trained AST-based transformer model
- Uses advanced preprocessing techniques including:
  - Spectral filtering
  - Log-mel spectrogram conversion
  - Zero-padding and normalization
- Threat Class Detection for Public Safety Surveillance
- Ranked Multi-Class Probabilities (Top-3 threat predictions)
- Modular Design – easy to plug in with CCTVs, IoT devices, or public infrastructure
- Can be extended to real-time streaming and edge devices
- Hosted on Hugging Face Deployment for global access and scalability
- Designed with MLOps-readiness in mind (supports future containerization and CI/CD pipelines)
  - Based on transformer architecture trained on real-world urban and environmental audio

---

## Tech Stack
- Python
- PyTorch
- Torchaudio
- Librosa
- Gradio
- Trained on ESC-50 and UrbanSound8K datasets (environmental and urban audio)

---

## Model Info
- Architecture: Audio Spectrogram Transformer (AST)
- Input: 5-second audio clips
- Output Classes: 
  - air_conditioner
  - car_horn
  - children_playing
  - dog_bark
  - drilling
  - engine_idling
  - gun_shot
  - jackhammer
  - siren
  - street_music

---

## Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/sonicshield.git
cd sonicshield

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## Example Usage
Upload an audio clip or record directly. The model returns the most probable threat label, such as:
```json
{
  "gun_shot": 0.96,
  "siren": 0.02,
  "dog_bark": 0.01
}
```

---

## Author
**Manishka Chawla**  

---

## Highlights
- Based on transformer architecture trained on real-world urban and environmental audio
- High-resolution audio representation through log-mel spectrograms and spectral filtering
- Evaluation-ready with ranked predictions and confidence scores
- Modular code structure for rapid integration with safety systems
- Hosted on Hugging Face Spaces for seamless deployment and sharing

---

"Empowering safety through sound."

---
