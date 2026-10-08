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

    waveform, sample_rate = torchaudio.load(
        audio_path
    )

    # Stereo -> Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    audio = waveform.numpy().flatten()

    return audio, sample_rate


# ============================================================
# Calculate Frame RMS Energy
# ============================================================

def calculate_frame_energy(
    audio,
    sample_rate,
    frame_duration=0.025,
    hop_duration=0.010
):

    frame_size = int(
        frame_duration * sample_rate
    )

    hop_size = int(
        hop_duration * sample_rate
    )

    energies = []
    times = []

    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):

        frame = audio[
            start:start + frame_size
        ]

        rms = np.sqrt(
            np.mean(frame ** 2)
        )

        energies.append(rms)

        times.append(
            start / sample_rate
        )

    return (
        np.array(energies),
        np.array(times)
    )


# ============================================================
# Detect Speech Region
# ============================================================

def detect_speech_region(
    audio,
    sample_rate
):

    energy, times = calculate_frame_energy(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    noise_level = np.percentile(
        energy,
        20
    )

    threshold = max(
        noise_level * 2.0,
        0.005
    )

    speech_frames = (
        energy > threshold
    )

    # --------------------------------------------------------
    # Find first and last speech frame
    # --------------------------------------------------------

    speech_indices = np.where(
        speech_frames
    )[0]

    if len(speech_indices) == 0:

        return None

    first_index = speech_indices[0]
    last_index = speech_indices[-1]

    start_time = times[
        first_index
    ]

    # Add frame duration to last frame
    end_time = (
        times[last_index] + 0.025
    )

    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "threshold": threshold
    }


# ============================================================
# Main Analysis
# ============================================================

def analyze_speech_region(
    audio_path
):

    print("Loading audio...")

    audio, sample_rate = load_audio(
        audio_path
    )

    total_duration = (
        len(audio) / sample_rate
    )

    speech_region = detect_speech_region(
        audio,
        sample_rate
    )

    print("\n========================================")
    print("        SPEECH REGION ANALYSIS")
    print("========================================")

    print(
        f"Total audio duration: "
        f"{total_duration:.2f} seconds"
    )

    if speech_region is None:

        print(
            "No speech region detected."
        )

        return None

    print(
        f"Speech starts at:     "
        f"{speech_region['start_time']:.2f} sec"
    )

    print(
        f"Speech ends at:       "
        f"{speech_region['end_time']:.2f} sec"
    )

    print(
        f"Speech duration:      "
        f"{speech_region['duration']:.2f} sec"
    )

    print(
        f"Detection threshold:  "
        f"{speech_region['threshold']:.4f}"
    )

    print("========================================")

    return speech_region


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    result = analyze_speech_region(
        AUDIO_PATH
    )