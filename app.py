# app.py (Gradio version for Hugging Face)
import torch
import torchaudio
import gradio as gr
from model import AudioTransformer
from preprocessing import preprocess_audio


# Define your classes (update if needed)
class_names = ['gunshot', 'scream', 'glass_break', 'background_noise']

# Load model
device = torch.device("cpu")
model = AudioTransformer(num_classes=len(class_names))
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict(audio_file):
    try:
        audio_tensor = preprocess_audio(audio_file)
        audio_tensor = audio_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(audio_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

        prediction = class_names[top_idx.item()]
        confidence = float(top_prob)

        result = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        return result
    except Exception as e:
        return {"error": str(e)}

# Gradio UI
audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
label_output = gr.Label(num_top_classes=3, label="Prediction")

demo = gr.Interface(
    fn=predict,
    inputs=audio_input,
    outputs=label_output,
    title="SonicShield - Crime Audio Detection",
    description="Upload an audio clip to detect crime-related sounds like gunshots or screams."
)

demo.launch()
