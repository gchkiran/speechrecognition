import numpy as np
import random
import librosa

def add_disturbance(audio, sr, disturbance_type=None):
    """
    Apply data augmentation to audio.
    
    Parameters:
        audio (np.ndarray): The audio signal.
        sr (int): Sample rate of the audio.
        disturbance_type (str): Type of augmentation ('noise', 'pitch_shift', 'high_pitch', 'time_stretch').
    
    Returns:
        np.ndarray: Augmented audio signal.
    """
    if disturbance_type is None:
        disturbance_type = random.choice(['noise', 'pitch_shift', 'high_pitch', 'time_stretch'])
    
    if disturbance_type == 'noise':
        noise_factor = random.uniform(0.005, 0.02)
        noise = np.random.randn(len(audio))
        audio_distorted = audio + noise_factor * noise
    
    elif disturbance_type == 'pitch_shift':
        pitch_shift = random.uniform(-4, 4)
        audio_distorted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
    
    elif disturbance_type == 'high_pitch':
        high_pitch_factor = random.uniform(0.001, 0.005)
        high_pitch_noise = np.sin(2 * np.pi * 5000 * np.arange(len(audio)) / sr)
        audio_distorted = audio + high_pitch_factor * high_pitch_noise
    
    elif disturbance_type == 'time_stretch':
        stretch_factor = random.uniform(0.8, 1.2)
        audio_distorted = librosa.effects.time_stretch(audio, rate=stretch_factor)
        if len(audio_distorted) > len(audio):
            audio_distorted = audio_distorted[:len(audio)]
        else:
            audio_distorted = np.pad(audio_distorted, (0, len(audio) - len(audio_distorted)))

    return audio_distorted