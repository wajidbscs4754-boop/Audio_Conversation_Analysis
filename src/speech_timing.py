import numpy as np

from syllable_detector import detect_syllables
from speech_region import load_audio, detect_speech_region


# ============================================================
# Configuration
# ============================================================

AUDIO_PATH = "data/syllable_test_new.wav"


# ============================================================
# Main Analysis
# ============================================================

def analyze_speech_timing(audio_path):

    print("Analyzing speech timing...")

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = load_audio(
        audio_path
    )

    # --------------------------------------------------------
    # Detect speech region
    # --------------------------------------------------------

    speech_region = detect_speech_region(
        audio,
        sample_rate
    )

    if speech_region is None:

        print("No speech region detected.")

        return None

    speech_start = speech_region["start_time"]
    speech_end = speech_region["end_time"]
    speech_duration = speech_region["duration"]

    # --------------------------------------------------------
    # Detect syllables
    # --------------------------------------------------------

    peak_times = detect_syllables(
        audio_path
    )

    # Keep only syllables inside speech region
    peak_times = peak_times[
        (peak_times >= speech_start)
        &
        (peak_times <= speech_end)
    ]

    syllable_count = len(
        peak_times
    )

    # --------------------------------------------------------
    # Calculate gaps
    # --------------------------------------------------------

    if syllable_count >= 2:

        gaps = np.diff(
            peak_times
        )

        average_gap = np.mean(
            gaps
        )

        gap_variation = np.std(
            gaps
        )

    else:

        average_gap = 0
        gap_variation = 0

    # --------------------------------------------------------
    # Syllables per second
    # --------------------------------------------------------

    if speech_duration > 0:

        syllables_per_sec = (
            syllable_count
            / speech_duration
        )

    else:

        syllables_per_sec = 0

    # ========================================================
    # Display Result
    # ========================================================

    print("\n")
    print("========================================")
    print("         SPEECH TIMING ANALYSIS")
    print("========================================")

    print(
        f"Speech start:       {speech_start:.2f} sec"
    )

    print(
        f"Speech end:         {speech_end:.2f} sec"
    )

    print(
        f"Speech duration:    {speech_duration:.2f} sec"
    )

    print(
        f"Syllable events:    {syllable_count}"
    )

    print(
        f"Average gap:        {average_gap:.2f} sec"
    )

    print(
        f"Gap variation:      {gap_variation:.2f} sec"
    )

    print(
        f"Syllables per second: "
        f"{syllables_per_sec:.2f}"
    )

    # --------------------------------------------------------
    # Syllable timings
    # --------------------------------------------------------

    print("\nSyllable timings:")

    for i, time in enumerate(
        peak_times
    ):

        print(
            f"  Syllable {i + 1}: "
            f"{time:.2f} sec"
        )

    print(
        "========================================"
    )

    return {

        "speech_start":
            speech_start,

        "speech_end":
            speech_end,

        "speech_duration":
            speech_duration,

        "syllable_count":
            syllable_count,

        "average_gap":
            average_gap,

        "gap_variation":
            gap_variation,

        "syllables_per_sec":
            syllables_per_sec,

        "syllable_times":
            peak_times.tolist()
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    result = analyze_speech_timing(
        AUDIO_PATH
    )