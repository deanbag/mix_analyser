import matchering as mg
import numpy as np
from flask import Flask, request, render_template, send_file
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
import psutil
from werkzeug.exceptions import RequestEntityTooLarge
from scipy.signal import resample, hilbert
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import pyloudnorm as pyln

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Enhanced logging to file
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Temporary store for results
latest_results = {}

def get_audio_duration(file_path):
    """Calculate audio duration."""
    try:
        audio, sr = sf.read(file_path)
        duration = len(audio) / sr
        logger.info(f"Duration for {file_path}: {duration:.1f} sec")
        return duration
    except Exception as e:
        logger.error(f"Error in get_audio_duration for {file_path}: {str(e)}")
        raise

def calculate_rms(audio):
    """Calculate RMS of audio."""
    try:
        rms = np.sqrt(np.mean(audio**2))
        return rms
    except Exception as e:
        logger.error(f"Error in calculate_rms: {str(e)}")
        raise

def get_average_rms(file_path, sr, chunk_duration=10):
    """Calculate average RMS over chunks."""
    try:
        audio, _ = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        chunk_samples = int(sr * chunk_duration)
        rms_values = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                rms = calculate_rms(chunk)
                rms_values.append(rms)
        avg_rms = np.mean(rms_values) if rms_values else 0
        logger.info(f"Average RMS for {file_path}: {avg_rms:.4f}")
        return avg_rms
    except Exception as e:
        logger.error(f"Error in get_average_rms for {file_path}: {str(e)}")
        raise

def calculate_lufs(file_path, sr):
    """Calculate LUFS."""
    try:
        audio, _ = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(audio)
        logger.info(f"LUFS for {file_path}: {lufs:.1f}")
        return lufs
    except Exception as e:
        logger.error(f"Error in calculate_lufs for {file_path}: {str(e)}")
        raise

def resample_audio(input_file, output_file, orig_sr, target_sr=44100):
    """Resample audio to target sample rate."""
    try:
        audio, sr = sf.read(input_file)
        if sr == target_sr:
            logger.info(f"No resampling needed: {input_file} is already at {sr} Hz")
            sf.write(output_file, audio, sr)
            return
        num_samples = int(len(audio) * target_sr / sr)
        if audio.ndim > 1:
            audio_resampled = np.zeros((num_samples, audio.shape[1]))
            for ch in range(audio.shape[1]):
                audio_resampled[:, ch] = resample(audio[:, ch], num_samples)
        else:
            audio_resampled = resample(audio, num_samples)
        sf.write(output_file, audio_resampled, target_sr)
        logger.info(f"Resampled {input_file} from {sr} Hz to {target_sr} Hz: {output_file}")
    except Exception as e:
        logger.error(f"Error in resample_audio for {input_file}: {str(e)}")
        raise

def extract_window(input_file, output_file, sr, start_time, duration=10):
    """Extract a time window from audio."""
    try:
        audio, sr = sf.read(input_file)
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        if start_sample >= len(audio):
            start_sample = max(0, len(audio) - int(duration * sr))
            logger.warning(f"Start time {start_time} sec is beyond file length; adjusted to {start_sample/sr:.1f} sec")
        if end_sample > len(audio):
            end_sample = len(audio)
            logger.warning(f"End time exceeds file length; adjusted to {end_sample/sr:.1f} sec")
        audio_window = audio[start_sample:end_sample]
        sf.write(output_file, audio_window, sr)
        logger.info(f"Extracted {duration}-second window from {start_time} sec: {output_file}")
        return audio_window
    except Exception as e:
        logger.error(f"Error in extract_window for {input_file}: {str(e)}")
        raise

def analyze_transients(audio, sr):
    """Analyze transient energy."""
    try:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        envelope = np.abs(hilbert(audio))
        envelope_diff = np.diff(envelope)
        transient_energy = np.mean(np.abs(envelope_diff))
        transient_energy_normalized = transient_energy * sr
        logger.info(f"Transient energy: {transient_energy_normalized:.4f}")
        return transient_energy_normalized
    except Exception as e:
        logger.error(f"Error in analyze_transients: {str(e)}")
        raise

def smooth_spectrum(data, window_size=50):
    """Smooth spectrum data."""
    try:
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        return smoothed
    except Exception as e:
        logger.error(f"Error in smooth_spectrum: {str(e)}")
        raise

def analyze_mix(target_file, reference_file):
    """Analyze target and reference audio files."""
    try:
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before analysis: {mem_before:.2f} MB")
        logger.info(f"Starting analysis: target={target_file}, reference={reference_file}")

        target_size = os.path.getsize(target_file) / 1024 / 1024
        ref_size = os.path.getsize(reference_file) / 1024 / 1024
        logger.info(f"File sizes: target={target_size:.2f} MB, reference={ref_size:.2f} MB")
        if target_size > 50 or ref_size > 50:
            raise ValueError("Each file must be under 50 MB.")

        target_duration = get_audio_duration(target_file)
        ref_duration = get_audio_duration(reference_file)
        logger.info(f"Durations: target={target_duration:.1f} sec, reference={ref_duration:.1f} sec")
        if target_duration > 300 or ref_duration > 300:
            raise ValueError("Files must be under 5 minutes in duration.")

        target_start_time = target_duration * 0.25
        ref_start_time = ref_duration * 0.25
        logger.info(f"Analysis window start times: target={target_start_time:.1f} sec, reference={ref_start_time:.1f} sec")

        _, sr = sf.read(target_file)

        logger.info("Extracting target audio window")
        target_snippet = os.path.join(UPLOAD_FOLDER, "target_snippet.wav")
        target_audio = extract_window(target_file, target_snippet, sr, target_start_time, duration=10)
        if target_audio.ndim > 1:
            target_audio = np.mean(target_audio, axis=1, keepdims=True).T
        logger.info(f"Target audio window loaded: shape={target_audio.shape}, sr={sr}")

        logger.info("Extracting reference audio window")
        reference_snippet = os.path.join(UPLOAD_FOLDER, "reference_snippet.wav")
        reference_audio = extract_window(reference_file, reference_snippet, sr, ref_start_time, duration=10)
        if reference_audio.ndim > 1:
            reference_audio = np.mean(reference_audio, axis=1, keepdims=True).T
        logger.info(f"Reference audio window loaded: shape={reference_audio.shape}, sr={sr}")

        target_snippet_resampled = os.path.join(UPLOAD_FOLDER, "target_snippet_resampled.wav")
        reference_snippet_resampled = os.path.join(UPLOAD_FOLDER, "reference_snippet_resampled.wav")
        resample_audio(target_snippet, target_snippet_resampled, orig_sr=sr, target_sr=44100)
        resample_audio(reference_snippet, reference_snippet_resampled, orig_sr=sr, target_sr=44100)

        target_audio, sr = sf.read(target_snippet_resampled)
        reference_audio, _ = sf.read(reference_snippet_resampled)

        mem_after_load = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after loading files: {mem_after_load:.2f} MB")

        target_window_rms = calculate_rms(target_audio)
        reference_window_rms = calculate_rms(reference_audio)

        target_avg_rms = get_average_rms(target_file, sr, chunk_duration=10)
        reference_avg_rms = get_average_rms(reference_file, sr, chunk_duration=10)
        logger.info(f"RMS values: target window={target_window_rms:.4f}, target avg={target_avg_rms:.4f}")
        logger.info(f"RMS values: reference window={reference_window_rms:.4f}, reference avg={reference_avg_rms:.4f}")

        target_lufs = calculate_lufs(target_snippet_resampled, sr)
        reference_lufs = calculate_lufs(reference_snippet_resampled, sr)
        lufs_diff_percent = float((reference_lufs - target_lufs) / abs(reference_lufs) * 100) if reference_lufs != 0 else 0
        logger.info(f"LUFS diff: target={target_lufs:.1f}, reference={reference_lufs:.1f}, diff={lufs_diff_percent:.1f}%")

        target_confidence = max(0, 100 - (abs(target_window_rms - target_avg_rms) / target_avg_rms * 100)) if target_avg_rms > 0 else 0
        reference_confidence = max(0, 100 - (abs(reference_window_rms - reference_avg_rms) / reference_avg_rms * 100)) if reference_avg_rms > 0 else 0
        confidence = (target_confidence + reference_confidence) / 2
        logger.info(f"Confidence metric: target={target_confidence:.1f}%, reference={reference_confidence:.1f}%, overall={confidence:.1f}%")

        logger.info("Converting to stereo")
        if target_audio.ndim == 1:
            target_audio = np.array([target_audio, target_audio]).T
        if reference_audio.ndim == 1:
            reference_audio = np.array([reference_audio, reference_audio]).T
        logger.info(f"Stereo shapes: target={target_audio.shape}, reference={reference_audio.shape}")

        if target_audio.shape[1] == 2:
            left_rms = calculate_rms(target_audio[:, 0])
            right_rms = calculate_rms(target_audio[:, 1])
            stereo_balance_diff = (left_rms - right_rms) / max(left_rms, right_rms) * 100 if max(left_rms, right_rms) > 0 else 0
        else:
            stereo_balance_diff = 0.0
        logger.info(f"Stereo balance difference: {stereo_balance_diff:.1f}%")

        if target_audio.shape[1] == 2:
            stereo_rms = calculate_rms(target_audio)
            mono_audio = (target_audio[:, 0] + target_audio[:, 1]) / 2
            mono_rms = calculate_rms(mono_audio)
            mono_compatibility_diff = (stereo_rms - mono_rms) / stereo_rms * 100 if stereo_rms > 0 else 0
        else:
            mono_compatibility_diff = 0.0
        logger.info(f"Mono compatibility difference: {mono_compatibility_diff:.1f}%")

        target_transient_energy = analyze_transients(target_audio, sr)
        reference_transient_energy = analyze_transients(reference_audio, sr)
        transient_diff = (target_transient_energy - reference_transient_energy) / reference_transient_energy * 100 if reference_transient_energy > 0 else 0
        logger.info(f"Transient difference: {transient_diff:.1f}%")

        target_rms = np.sqrt(np.mean(target_audio**2))
        reference_rms = np.sqrt(np.mean(reference_audio**2))
        rms_diff_percent = float((reference_rms - target_rms) / reference_rms * 100)

        target_peak = np.max(np.abs(target_audio))
        reference_peak = np.max(np.abs(reference_audio))
        peak_diff_percent = float((reference_peak - target_peak) / reference_peak * 100)
        target_dr = float(20 * np.log10(target_peak / target_rms)) if target_rms > 0 else 0

        target_spec = np.abs(np.fft.rfft(target_audio.mean(axis=1)))
        reference_spec = np.abs(np.fft.rfft(reference_audio.mean(axis=1)))
        freqs = np.fft.rfftfreq(len(target_audio), 1/sr)
        logger.info(f"Frequency bins: min={freqs[0]:.1f} Hz, max={freqs[-1]:.1f} Hz")

        def get_band_energy(spec, freqs, low, high):
            try:
                mask = (freqs >= low) & (freqs <= high)
                if not np.any(mask):
                    logger.warning(f"No frequencies found in range {low}-{high} Hz")
                    return 0.0
                band_spec = spec[mask]
                energy = np.mean(band_spec)
                return energy
            except Exception as e:
                logger.error(f"Error in get_band_energy for {low}-{high} Hz: {str(e)}")
                raise

        bass_range, mid_range, treble_range = (20, 250), (250, 4000), (4000, 20000)
        target_bass = get_band_energy(target_spec, freqs, *bass_range)
        target_mids = get_band_energy(target_spec, freqs, *mid_range)
        target_treble = get_band_energy(target_spec, freqs, *treble_range)
        reference_bass = get_band_energy(reference_spec, freqs, *bass_range)
        reference_mids = get_band_energy(reference_spec, freqs, *mid_range)
        reference_treble = get_band_energy(reference_spec, freqs, *treble_range)

        bass_diff = float((reference_bass - target_bass) / reference_bass * 100) if reference_bass > 0 else 0
        mids_diff = float((reference_mids - target_mids) / reference_mids * 100) if reference_mids > 0 else 0
        treble_diff = float((reference_treble - target_treble) / reference_treble * 100) if reference_treble > 0 else 0
        logger.info(f"Spectral differences: bass={bass_diff:.1f}%, mids={mids_diff:.1f}%, treble={treble_diff:.1f}%")

        target_stereo_diff = np.mean(np.abs(target_audio[:, 0] - target_audio[:, 1]))
        reference_stereo_diff = np.mean(np.abs(reference_audio[:, 0] - reference_audio[:, 1]))
        stereo_diff_percent = 0.0
        if abs(reference_stereo_diff) > 1e-10:
            stereo_diff_percent = float((reference_stereo_diff - target_stereo_diff) / reference_stereo_diff * 100)
        logger.info(f"Stereo diff calculated: {stereo_diff_percent:.1f}%")

        logger.info("Generating spectrum plot")
        target_spec_db = 20 * np.log10(target_spec + 1e-6)
        reference_spec_db = 20 * np.log10(reference_spec + 1e-6)
        target_spec_db = smooth_spectrum(target_spec_db, window_size=50)
        reference_spec_db = smooth_spectrum(reference_spec_db, window_size=50)
        max_target_db = np.max(target_spec_db)
        max_reference_db = np.max(reference_spec_db)
        max_db = max(max_target_db, max_reference_db)
        if max_db > -np.inf:
            target_spec_db = target_spec_db - max_db
            reference_spec_db = reference_spec_db - max_db
        min_db = min(np.min(target_spec_db), np.min(reference_spec_db))
        min_db = max(min_db, -120)
        max_db = 0
        buffer_db = 10
        plt.figure(figsize=(10, 4))
        plt.semilogx(freqs, target_spec_db, label="Target", color="#3498db")
        plt.semilogx(freqs, reference_spec_db, label="Reference", color="#e74c3c")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("Frequency Spectrum Comparison")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.2)
        plt.xlim(20, 20000)
        plt.ylim(min_db - buffer_db, max_db + buffer_db)
        plot_path = os.path.join(STATIC_FOLDER, "spectrum.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Spectrum plot saved: {plot_path}, size={os.path.getsize(plot_path)} bytes")

        feedback = []

        if lufs_diff_percent > 15:
            feedback.append({
                "category": "Loudness",
                "text": f"Your mix is {lufs_diff_percent:.1f}% quieter than the reference ({target_lufs:.1f} LUFS vs {reference_lufs:.1f} LUFS). Apply a limiter to match perceived loudness."
            })
        elif lufs_diff_percent < -15:
            feedback.append({
                "category": "Loudness",
                "text": f"Your mix is {abs(lufs_diff_percent):.1f}% louder than the reference ({target_lufs:.1f} LUFS vs {reference_lufs:.1f} LUFS). Apply a limiter to match perceived loudness."
            })

        if target_peak > 0.99:
            feedback.append({
                "category": "Peak Level",
                "text": "Peak level is very high. Reduce the gain by 1-2 dB to prevent clipping."
            })
        elif peak_diff_percent > 20:
            feedback.append({
                "category": "Peak Level",
                "text": f"Peak level is {peak_diff_percent:.1f}% lower than the reference. Increase the gain by 1-2 dB to match the reference's headroom."
            })
        elif peak_diff_percent < -20:
            feedback.append({
                "category": "Peak Level",
                "text": f"Peak level is {abs(peak_diff_percent):.1f}% higher than the reference. Reduce the gain by 1-2 dB to avoid clipping."
            })

        if bass_diff > 20:
            feedback.append({
                "category": "Bass",
                "text": f"Bass (20-250 Hz) is {bass_diff:.1f}% weaker than the reference. Boost 60-100 Hz by 2-4 dB."
            })
        elif bass_diff < -20:
            feedback.append({
                "category": "Bass",
                "text": f"Bass (20-250 Hz) is {abs(bass_diff):.1f}% too strong. Cut 60-100 Hz by 2-4 dB."
            })

        if mids_diff > 20:
            feedback.append({
                "category": "Mids",
                "text": f"Mids (250 Hz-4 kHz) are {mids_diff:.1f}% weaker than the reference. Boost 1-2 kHz by 1-3 dB."
            })
        elif mids_diff < -20:
            feedback.append({
                "category": "Mids",
                "text": f"Mids (250 Hz-4 kHz) are {abs(mids_diff):.1f}% too strong. Cut 1-2 kHz by 1-3 dB."
            })

        if treble_diff > 20:
            feedback.append({
                "category": "Treble",
                "text": f"Treble (4-20 kHz) is {treble_diff:.1f}% weaker than the reference. Apply a high shelf at 8 kHz, +2-4 dB."
            })
        elif treble_diff < -20:
            feedback.append({
                "category": "Treble",
                "text": f"Treble (4-20 kHz) is {abs(treble_diff):.1f}% too strong. Apply a high shelf at 8 kHz, -2-4 dB."
            })

        if stereo_diff_percent > 20:
            feedback.append({
                "category": "Stereo Width",
                "text": f"Stereo width is {stereo_diff_percent:.1f}% narrower than the reference. Use a stereo widener plugin to increase width by 10-20%."
            })
        elif stereo_diff_percent < -20:
            feedback.append({
                "category": "Stereo Width",
                "text": f"Stereo width is {abs(stereo_diff_percent):.1f}% wider than the reference. Reduce stereo width on elements like reverb or delay by 10-20%."
            })

        if abs(stereo_balance_diff) > 5:
            if stereo_balance_diff > 0:
                feedback.append({
                    "category": "Stereo Balance",
                    "text": f"Left channel is {stereo_balance_diff:.1f}% louder than the right. Adjust panning or reduce the left channel gain by 1-2 dB to balance the stereo image."
                })
            else:
                feedback.append({
                    "category": "Stereo Balance",
                    "text": f"Right channel is {abs(stereo_balance_diff):.1f}% louder than the left. Adjust panning or reduce the right channel gain by 1-2 dB to balance the stereo image."
                })

        if target_dr < 6:
            feedback.append({
                "category": "Dynamic Range",
                "text": f"Dynamic range is low ({target_dr:.1f} dB). Reduce compression by setting the ratio to 2:1-3:1 and increase the threshold by 3-5 dB."
            })
        elif target_dr > 12:
            feedback.append({
                "category": "Dynamic Range",
                "text": f"Dynamic range is high ({target_dr:.1f} dB). Apply light compression with a ratio of 2:1-3:1 and threshold around -20 to -15 dB."
            })

        if transient_diff > 20:
            feedback.append({
                "category": "Transients",
                "text": f"Transients are {transient_diff:.1f}% sharper than the reference. Apply a compressor with a fast attack (1-5 ms) and medium release (50-100 ms)."
            })
        elif transient_diff < -20:
            feedback.append({
                "category": "Transients",
                "text": f"Transients are {abs(transient_diff):.1f}% softer than the reference. Use a transient shaper to increase attack by 5-10 dB."
            })

        if mono_compatibility_diff > 10:
            feedback.append({
                "category": "Mono Compatibility",
                "text": f"Your mix loses {mono_compatibility_diff:.1f}% level when summed to mono, indicating phase cancellation issues. Check for out-of-phase stereo effects and reduce their width by 10-20%, then listen in mono to identify problem areas."
            })

        if not feedback:
            feedback.append({
                "category": "General",
                "text": "Your mix is well-balanced! Consider subtle adjustments to taste."
            })

        logger.info("Generated feedback")

        logger.info("Starting Matchering for preview")
        logger.info(f"RESAMPY_DISABLE_NUMBA is set to: {os.environ.get('RESAMPY_DISABLE_NUMBA')}")
        output_file = os.path.join(OUTPUT_FOLDER, "output.wav")
        results = [mg.pcm24(output_file)]
        mg.process(target=target_snippet_resampled, reference=reference_snippet_resampled, results=results)
        logger.info(f"Matchering completed: {output_file}")

        logger.info("Generating audio previews")
        for name, path in [("target", target_snippet_resampled), ("reference", reference_snippet_resampled), ("mastered", output_file)]:
            audio = AudioSegment.from_wav(path)
            mp3_path = os.path.join(STATIC_FOLDER, f"{name}.mp3")
            audio.export(mp3_path, format="mp3")

        mem_after_processing = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after processing: {mem_after_processing:.2f} MB")

        # Clean up temporary files
        for temp_file in [target_snippet, reference_snippet, target_snippet_resampled, reference_snippet_resampled, output_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error removing temporary file {temp_file}: {str(e)}")

        dr_percentage = min((target_dr / 30.0) * 100, 100)

        mem_final = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after cleanup: {mem_final:.2f} MB")

        results_dict = {
            "rms_diff_percent": rms_diff_percent,
            "bass_diff_percent": bass_diff,
            "mids_diff_percent": mids_diff,
            "treble_diff_percent": treble_diff,
            "stereo_diff_percent": stereo_diff_percent,
            "peak_diff_percent": peak_diff_percent,
            "dynamic_range": target_dr,
            "dr_percentage": dr_percentage,
            "confidence": confidence,
            "feedback": feedback,
            "spectrum_plot": "/static/spectrum.png",
            "target_audio": "/static/target.mp3",
            "reference_audio": "/static/reference.mp3",
            "mastered_audio": "/static/mastered.mp3",
            "rms_diff_width": min(abs(rms_diff_percent), 100),
            "bass_diff_width": min(abs(bass_diff), 100),
            "mids_diff_width": min(abs(mids_diff), 100),
            "treble_diff_width": min(abs(treble_diff), 100),
            "stereo_diff_width": min(abs(stereo_diff_percent), 100),
            "peak_diff_width": min(abs(peak_diff_percent), 100),
            "stereo_balance_diff": stereo_balance_diff,
            "stereo_balance_width": min(abs(stereo_balance_diff), 100),
            "transient_diff": transient_diff,
            "transient_diff_width": min(abs(transient_diff), 100),
            "mono_compatibility_diff": mono_compatibility_diff,
            "mono_compatibility_width": min(abs(mono_compatibility_diff), 100),
            "lufs_diff_percent": lufs_diff_percent,
            "lufs_diff_width": min(abs(lufs_diff_percent), 100),
            "target_lufs": target_lufs,
            "reference_lufs": reference_lufs
        }
        return results_dict
    except MemoryError as e:
        logger.error(f"MemoryError during analysis: {str(e)}")
        return {"status": "error", "message": "Out of memory. Try smaller files or contact support."}
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}

@app.route("/", methods=["GET", "POST"])
def upload():
    global latest_results
    if request.method == "POST":
        try:
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage before upload: {mem_before:.2f} MB")

            if "target" not in request.files or "reference" not in request.files:
                return render_template("upload.html", error="Please upload both target and reference files.")
            
            target = request.files["target"]
            reference = request.files["reference"]
            
            if target.filename == "" or reference.filename == "":
                return render_template("upload.html", error="No file selected.")
            
            if not (target.filename.endswith(('.wav', '.mp3')) and reference.filename.endswith(('.wav', '.mp3'))):
                return render_template("upload.html", error="Only WAV or MP3 files are supported.")

            target_path = os.path.join(UPLOAD_FOLDER, "target.wav")
            reference_path = os.path.join(UPLOAD_FOLDER, "reference.wav")
            
            target.save(target_path)
            reference.save(reference_path)
            
            results = analyze_mix(target_path, reference_path)
            if isinstance(results, dict) and results.get("status") == "error":
                return render_template("upload.html", error=results["message"])

            latest_results = results
            for temp_path in [target_path, reference_path]:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logger.info(f"Removed upload file: {temp_path}")
                    except Exception as e:
                        logger.error(f"Error removing upload file {temp_path}: {str(e)}")

            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage after upload: {mem_after:.2f} MB")

            return render_template("results.html", results=results)
        except RequestEntityTooLarge:
            return render_template("upload.html", error="Files are too large. The total upload size must be under 100 MB (each file under 50 MB).")
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return render_template("upload.html", error=f"Upload failed: {str(e)}")
    return render_template("upload.html")

@app.route("/download_report")
def download_report():
    global latest_results
    try:
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before download_report: {mem_before:.2f} MB")

        if not latest_results:
            return render_template("upload.html", error="No analysis results available. Please analyze a mix first.")

        pdf_path = os.path.join(OUTPUT_FOLDER, "mix_analysis_report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        logo_path = os.path.join(STATIC_FOLDER, "logo.png")
        if os.path.exists(logo_path):
            logo = Image(logo_path, width=2*inch, height=2*inch*170/924)
            elements.append(logo)
            elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Mix Analysis Report", styles['Title']))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Feedback", styles['Heading2']))
        for item in latest_results['feedback']:
            elements.append(Paragraph(f"<b>{item['category']}:</b> {item['text']}", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Metrics", styles['Heading2']))
        metrics_data = [
            ["Metric", "Value"],
            ["Confidence Metric", f"{latest_results['confidence']:.1f}%"],
            ["LUFS Difference", f"{latest_results['lufs_diff_percent']:.1f}%"],
            ["RMS Difference", f"{latest_results['rms_diff_percent']:.1f}%"],
            ["Bass Difference", f"{latest_results['bass_diff_percent']:.1f}%"],
            ["Mids Difference", f"{latest_results['mids_diff_percent']:.1f}%"],
            ["Treble Difference", f"{latest_results['treble_diff_percent']:.1f}%"],
            ["Stereo Width Difference", f"{latest_results['stereo_diff_percent']:.1f}%"],
            ["Stereo Balance Difference", f"{latest_results['stereo_balance_diff']:.1f}%"],
            ["Transient Difference", f"{latest_results['transient_diff']:.1f}%"],
            ["Peak Difference", f"{latest_results['peak_diff_percent']:.1f}%"],
            ["Dynamic Range", f"{latest_results['dynamic_range']:.1f} dB"],
            ["Mono Compatibility", f"{latest_results['mono_compatibility_diff']:.1f}%"]
        ]
        table = Table(metrics_data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), '#3498db'),
            ('TEXTCOLOR', (0, 0), (-1, 0), '#ffffff'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), '#f9f9f9'),
            ('GRID', (0, 0), (-1, -1), 1, '#e0e0e0')
        ])
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Frequency Spectrum", styles['Heading2']))
        plot_path = os.path.join(STATIC_FOLDER, "spectrum.png")
        if os.path.exists(plot_path):
            img = Image(plot_path, width=6*inch, height=2.4*inch)
            elements.append(img)

        doc.build(elements)

        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after download_report: {mem_after:.2f} MB")

        return send_file(pdf_path, as_attachment=True, download_name="mix_analysis_report.pdf")
    except Exception as e:
        logger.error(f"Error in download_report: {str(e)}")
        return render_template("upload.html", error=f"Report generation failed: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
