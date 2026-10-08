import numpy as np
import torchaudio
from scipy.signal import find_peaks


# ============================================================
# Configuration
# ============================================================

AUDIO_PATH = "data/computer_test.wav"
# Target word
TARGET_WORD = "computer"

# creator = 3 syllables
TARGET_SYLLABLES = 3


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
# Calculate RMS Energy
# ============================================================

def calculate_energy(audio, sample_rate):

    frame_size = int(0.025 * sample_rate)
    hop_size = int(0.010 * sample_rate)

    energy = []

    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):

        frame = audio[start:start + frame_size]

        rms = np.sqrt(
            np.mean(frame ** 2)
        )

        energy.append(rms)

    energy = np.array(energy)

    return energy, hop_size


# ============================================================
# Smooth Signal
# ============================================================

def smooth_signal(signal, window_size=9):

    kernel = np.ones(window_size) / window_size

    return np.convolve(
        signal,
        kernel,
        mode="same"
    )


# ============================================================
# Detect Syllables
# ============================================================

def detect_syllables(audio_path):

    print("Loading audio...")

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = load_audio(
        audio_path
    )

    duration = len(audio) / sample_rate

    print(
        f"Sample rate: {sample_rate} Hz"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    # --------------------------------------------------------
    # 1. Calculate Energy
    # --------------------------------------------------------

    energy, hop_size = calculate_energy(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # Normalize Energy
    # --------------------------------------------------------

    if np.max(energy) > 0:

        energy = energy / np.max(energy)

    # --------------------------------------------------------
    # 2. Smooth Energy
    # --------------------------------------------------------

    smooth_energy = smooth_signal(
        energy,
        window_size=9
    )

    # --------------------------------------------------------
    # 3. Adaptive Threshold
    # --------------------------------------------------------

    noise_level = np.percentile(
        smooth_energy,
        20
    )

    threshold = max(
        noise_level * 2.5,
        0.08
    )

    # --------------------------------------------------------
    # 4. Detect Syllable Candidates
    # --------------------------------------------------------

    peaks, properties = find_peaks(

        smooth_energy,

        # Minimum distance between peaks
        distance=int(
            0.18 / 0.010
        ),

        # Peak prominence
        prominence=0.04,

        # Minimum height
        height=threshold
    )

    peak_times = (
        peaks * hop_size / sample_rate
    )

    # --------------------------------------------------------
    # 5. Merge Very Close Peaks
    # --------------------------------------------------------

    merged_peaks = []

    minimum_gap = 0.15

    for peak in peak_times:

        if not merged_peaks:

            merged_peaks.append(
                peak
            )

            continue

        if (
            peak - merged_peaks[-1]
            >= minimum_gap
        ):

            merged_peaks.append(
                peak
            )

    peak_times = np.array(
        merged_peaks
    )

    # --------------------------------------------------------
    # Detected Syllable Count
    # --------------------------------------------------------

    detected_count = len(
        peak_times
    )

    # ========================================================
    # Syllable Match
    # ========================================================

    if detected_count == TARGET_SYLLABLES:

        syllable_match = True

    else:

        syllable_match = False

    # ========================================================
    # Speech Rate
    # ========================================================

    if duration > 0:

        syllables_per_sec = (
            detected_count / duration
        )

    else:

        syllables_per_sec = 0

    # ========================================================
    # Gap Variation / Rhythm
    # ========================================================

    if len(peak_times) >= 2:

        gaps = np.diff(
            peak_times
        )

        gap_variation = np.std(
            gaps
        )

    else:

        gap_variation = 0

    # ========================================================
    # Student-Friendly Result
    # ========================================================

    print("\n")

    print(
        "========================================"
    )

    print(
        "         STONE YLLABLE STONES"
    )

    print(
        "========================================"
    )

    print(
        f"Word: {TARGET_WORD}"
    )

    print(
        f"Target syllables: {TARGET_SYLLABLES}"
    )

    print(
        f"Your syllables: {detected_count}"
    )

    # --------------------------------------------------------
    # Stones
    # --------------------------------------------------------

    print("\nStones:")

    if detected_count == 0:

        print(
            "  No stones detected."
        )

    else:

        for i in range(
            detected_count
        ):

            print(
                f"  STONE Stone {i + 1}"
            )

    # --------------------------------------------------------
    # Syllable Match
    # --------------------------------------------------------

    print("\nSyllable Match:")

    if syllable_match:

        print(
            "  PASS - Great! All syllables detected."
        )

    else:

        print(
            "  FAIL - Try saying the word again."
        )

    # --------------------------------------------------------
    # Speech Rate
    # --------------------------------------------------------

    print("\nSpeech Rate:")

    print(
        f"  {syllables_per_sec:.2f} "
        "syllables/second"
    )

    # --------------------------------------------------------
    # Rhythm
    # --------------------------------------------------------

    print("\nRhythm:")

    print(
        f"  Gap variation: "
        f"{gap_variation:.3f} seconds"
    )

    if gap_variation < 0.15:

        print(
            "  Rhythm: Very steady"
        )

    elif gap_variation < 0.30:

        print(
            "  Rhythm: Good"
        )

    else:

        print(
            "  Rhythm: Needs practice"
        )

    print(
        "========================================"
    )

    # ========================================================
    # Return Results
    # ========================================================

    return {

        "target_word": TARGET_WORD,

        "target_syllables":
            TARGET_SYLLABLES,

        "detected_syllables":
            detected_count,

        "syllable_match":
            syllable_match,

        "syllables_per_sec":
            syllables_per_sec,

        "gap_variation":
            gap_variation
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    result = detect_syllables(
        AUDIO_PATH
    )