import matchering as mg
import numpy as np
from flask import Flask, request, render_template, send_file
import os
import logging
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_mix(target_file, reference_file):
    logger.info(f"Starting analysis: target={target_file}, reference={reference_file}")
    
    target_size = os.path.getsize(target_file) / 1024 / 1024
    ref_size = os.path.getsize(reference_file) / 1024 / 1024
    logger.info(f"File sizes: target={target_size:.2f} MB, reference={ref_size:.2f} MB")
    if target_size > 2 or ref_size > 2:
        raise ValueError("Files must be under 2 MB each.")
    
    logger.info("Loading target audio")
    target_audio, sr = sf.read(target_file)
    target_audio = target_audio[:int(sr * 10)]  # Limit to 10s
    if target_audio.ndim == 2:
        target_audio = target_audio.T  # Transpose to channels-first
    logger.info(f"Target audio loaded: shape={target_audio.shape}, sr={sr}")
    
    logger.info("Loading reference audio")
    reference_audio, _ = sf.read(reference_file)
    reference_audio = reference_audio[:int(sr * 10)]
    if reference_audio.ndim == 2:
        reference_audio = reference_audio.T
    logger.info(f"Reference audio loaded: shape={reference_audio.shape}, sr={sr}")
    
    logger.info("Converting to stereo")
    if target_audio.ndim == 1:
        target_audio = np.array([target_audio, target_audio])
    if reference_audio.ndim == 1:
        reference_audio = np.array([reference_audio, reference_audio])
    logger.info(f"Stereo shapes: target={target_audio.shape}, reference={reference_audio.shape}")
    
    target_rms = np.sqrt(np.mean(target_audio**2))
    reference_rms = np.sqrt(np.mean(reference_audio**2))
    rms_diff_percent = (reference_rms - target_rms) / reference_rms * 100
    
    target_spec = np.abs(np.fft.rfft(target_audio.mean(axis=0)))
    reference_spec = np.abs(np.fft.rfft(reference_audio.mean(axis=0)))
    freqs = np.fft.rfftfreq(len(target_audio.mean(axis=0)), 1/sr)
    logger.info("Computed spectra")
    
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
        feedback.append("Your mix is quieter than the reference. Use a compressor or limiter.")
    elif rms_diff_percent < -15:
        feedback.append("Your mix is louder than the reference. Reduce gain to avoid distortion.")
    if bass_diff > 20:
        feedback.append("Your mix lacks bass. Boost frequencies around 20-250 Hz.")
    
    logger.info("Generated feedback")
    del target_audio, reference_audio, target_spec, reference_spec
    
    output_file = os.path.join(OUTPUT_FOLDER, "output.wav")
    logger.info("Starting Matchering")
    results = [mg.pcm24(output_file)]
    mg.process(target=target_file, reference=reference_file, results=results)
    logger.info(f"Matchering completed: {output_file}")
    
    results_dict = {
        "rms_diff_percent": rms_diff_percent,
        "bass_diff_percent": bass_diff,
        "mids_diff_percent": mids_diff,
        "treble_diff_percent": treble_diff,
        "stereo_diff_percent": stereo_diff_percent,
        "peak_diff_percent": peak_diff_percent,
        "feedback": "\n".join(feedback) if feedback else "Your mix is well-balanced!",
        "mastered_output": output_file
    }
    return results_dict

# [Rest of your routes unchanged...]
