import numpy as np
import torchaudio


# ============================================================
# Configuration
# ============================================================

AUDIO_PATH = "data/computer_test.wav"


# ============================================================
# Load Audio
# ============================================================

def load_audio(audio_path):

    waveform, sample_rate = torchaudio.load(audio_path)

    # Stereo -> Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    audio = waveform.numpy().flatten()

    return audio, sample_rate


# ============================================================
# Calculate RMS Loudness
# ============================================================

def calculate_loudness(audio):

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    return rms


# ============================================================
# Calculate Peak Amplitude
# ============================================================

def calculate_peak_amplitude(audio):

    peak = np.max(
        np.abs(audio)
    )

    return peak


# ============================================================
# Calculate Zero Crossing Rate
# ============================================================

def calculate_zero_crossing_rate(audio):

    zero_crossings = np.sum(
        np.abs(
            np.diff(
                np.sign(audio)
            )
        ) > 0
    )

    zcr = zero_crossings / len(audio)

    return zcr


# ============================================================
# Detect Active Speech
# ============================================================

def detect_speech_activity(audio):

    threshold = 0.02

    active_samples = np.abs(audio) > threshold

    speech_samples = np.sum(
        active_samples
    )

    speech_ratio = (
        speech_samples / len(audio)
    )

    return speech_ratio


# ============================================================
# Analyze Audio
# ============================================================

def analyze_pronunciation_features(audio_path):

    print("Loading audio...")

    audio, sample_rate = load_audio(
        audio_path
    )

    duration = (
        len(audio) / sample_rate
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    loudness = calculate_loudness(
        audio
    )

    peak_amplitude = calculate_peak_amplitude(
        audio
    )

    zero_crossing_rate = calculate_zero_crossing_rate(
        audio
    )

    speech_activity = detect_speech_activity(
        audio
    )

    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    print("\n========================================")
    print("     PRONUNCIATION AUDIO FEATURES")
    print("========================================")

    print(
        f"Sample rate:          {sample_rate} Hz"
    )

    print(
        f"Duration:             {duration:.2f} seconds"
    )

    print(
        f"Loudness (RMS):       {loudness:.4f}"
    )

    print(
        f"Peak amplitude:       {peak_amplitude:.4f}"
    )

    print(
        f"Zero crossing rate:   {zero_crossing_rate:.4f}"
    )

    print(
        f"Speech activity:      {speech_activity * 100:.2f}%"
    )

    print("========================================")

    return {
        "duration": duration,
        "loudness": loudness,
        "peak_amplitude": peak_amplitude,
        "zero_crossing_rate": zero_crossing_rate,
        "speech_activity": speech_activity
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    result = analyze_pronunciation_features(
        AUDIO_PATH
    )