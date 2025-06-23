import librosa
import numpy as np
import torch

def preprocess_audio(path, sample_rate=16000, n_mels=128):
    """
    Load an audio file and convert it into a normalized mel spectrogram tensor.

    Args:
        path (str): Path to the audio file.
        sample_rate (int): Target sampling rate for the audio.
        n_mels (int): Number of mel bands to generate.

    Returns:
        torch.Tensor: Tensor of shape [1, 1, H, W] suitable for model input.
    """
    # Load audio file
    y, sr = librosa.load(path, sr=sample_rate)

    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # Convert to decibels
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to 0-1
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # Convert to tensor: [1, 1, H, W]
    mel_tensor = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).float()

    return mel_tensor
