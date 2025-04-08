import matchering as mg
import librosa
import numpy as np
from flask import Flask, request, render_template, send_file
import os
import logging
import psutil  # Add for memory tracking

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Memory usage: {mem:.2f} MB")

def analyze_mix(target_file, reference_file):
    logger.info(f"Starting analysis: target={target_file}, reference={reference_file}")
    log_memory()
    
    # Check file sizes
    target_size = os.path.getsize(target_file) / 1024 / 1024  # MB
    ref_size = os.path.getsize(reference_file) / 1024 / 1024  # MB
    logger.info(f"File sizes: target={target_size:.2f} MB, reference={ref_size:.2f} MB")
    if target_size > 2 or ref_size > 2:
        raise ValueError("Files must be under 2 MB each.")
    
    # Load audio
    logger.info("Loading audio files")
    target_audio, sr = librosa.load(target_file, sr=22050, mono=False, duration=15)
    reference_audio, _ = librosa.load(reference_file, sr=22050, mono=False, duration=15)
    logger.info(f"Loaded audio: target shape={target_audio.shape}, sr={sr}")
    log_memory()
    
    if target_audio.ndim == 1:
        target_audio = np.array([target_audio, target_audio])
    if reference_audio.ndim == 1:
        reference_audio = np.array([reference_audio, reference_audio])
    
    target_rms = np.sqrt(np.mean(target_audio**2))
    reference_rms = np.sqrt(np.mean(reference_audio**2))
    rms_diff_percent = (reference_rms - target_rms) / reference_rms * 100
    
    target_spec = np.abs(librosa.stft(target_audio.mean(axis=0)))
    reference_spec = np.abs(librosa.stft(reference_audio.mean(axis=0)))
    freqs = librosa.fft_frequencies(sr=sr)
    logger.info("Computed spectra")
    log_memory()
    
    def get_band_energy(spec, freqs, low, high):
        mask = (freqs >= low) & (freqs <= high)
        return np.mean(spec[mask])
    
    bass_range, mid_range, treble_range = (20, 250), (250, 4000), (4000, 20000)
    target_bass = get_band_energy(target_spec, freqs, *bass_range)
    target_mids = get_band_energy(target_spec, freqs, *mid_range)
    target_treble = get_band_energy(target_spec, freqs, *treble_range)
    reference_bass = get_band_energy(reference_spec, freqs, *bass_range)
    reference_mids = get_band_energy(reference_spec, freqs, *mid_range)
    reference_treble = get_band_energy(reference_spec, freqs, *treble_range)
    
    bass_diff = (reference_bass - target_bass) / reference_bass * 100
    mids_diff = (reference_mids - target_mids) / reference_mids * 100
    treble_diff = (reference_treble - target_treble) / reference_treble * 100
    
    target_stereo_diff = np.mean(np.abs(target_audio[0] - target_audio[1]))
    reference_stereo_diff = np.mean(np.abs(reference_audio[0] - reference_audio[1]))
    stereo_diff_percent = (reference_stereo_diff - target_stereo_diff) / reference_stereo_diff * 100
    
    target_peak = np.max(np.abs(target_audio))
    reference_peak = np.max(np.abs(reference_audio))
    peak_diff_percent = (reference_peak - target_peak) / reference_peak * 100
    
    feedback = []
    if rms_diff_percent > 15:
        feedback.append("Your mix is quieter than the reference. Use a compressor or limiter to increase loudness.")
    # [Add your other feedback logic...]
    
    logger.info("Generated feedback")
    del target_audio, reference_audio, target_spec, reference_spec
    log_memory()
    
    output_file = os.path.join(OUTPUT_FOLDER, "output.wav")
    logger.info("Starting Matchering")
    results = [mg.pcm24(output_file)]
    mg.process(target=target_file, reference=reference_file, results=results)
    logger.info(f"Matchering completed: {output_file}")
    log_memory()
    
    results_dict = {
        "rms_diff_percent": rms_diff_percent,
        "bass_diff_percent": bass_diff,
        "mids_diff_percent": mids_diff,
        "treble_diff_percent": treble_diff,
        "stereo_diff_percent": stereo_diff_percent,
        "peak_diff_percent": peak_diff_percent,
        "feedback": "\n".join(feedback) if feedback else "Your mix is well-balanced with the reference!",
        "mastered_output": output_file
    }
    return results_dict

# [Rest of your routes remain unchanged...]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
