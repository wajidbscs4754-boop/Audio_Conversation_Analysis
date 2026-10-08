import re
import whisper
import numpy as np
import soundfile as sf


# ============================================================
# SETTINGS
# ============================================================

AUDIO_PATH = "data/computer_test.wav"
TARGET_WORD = "computer"

TEMP_AUDIO = "data/_word_speech.wav"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Convert text into a simple comparable form.
    """

    text = text.lower().strip()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein_distance(a, b):
    """
    Calculate character-level edit distance.
    """

    rows = len(a) + 1
    cols = len(b) + 1

    dp = [[0] * cols for _ in range(rows)]

    # First column
    for i in range(rows):
        dp[i][0] = i

    # First row
    for j in range(cols):
        dp[0][j] = j

    # Fill table
    for i in range(1, rows):

        for j in range(1, cols):

            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


# ============================================================
# SIMILARITY SCORE
# ============================================================

def similarity_score(target, recognized):
    """
    Character-level text similarity.

    IMPORTANT:
    This is only a word-recognition signal.
    It is NOT pronunciation accuracy.
    """

    if not target:
        return 0.0

    if not recognized:
        return 0.0

    distance = levenshtein_distance(
        target,
        recognized
    )

    maximum_length = max(
        len(target),
        len(recognized)
    )

    return 1 - (
        distance / maximum_length
    )


# ============================================================
# SPEECH REGION DETECTION
# ============================================================

def find_speech_region(audio, sample_rate):
    """
    Detect the approximate region containing speech
    using RMS energy.
    """

    # Convert stereo audio to mono
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    # Frame settings
    frame_size = int(
        0.025 * sample_rate
    )

    hop_size = int(
        0.010 * sample_rate
    )

    energies = []

    # Calculate RMS energy for each frame
    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):

        frame = audio[
            start:start + frame_size
        ]

        rms = np.sqrt(
            np.mean(frame ** 2) + 1e-10
        )

        energies.append(rms)

    energies = np.array(
        energies
    )

    # If audio is too short
    if len(energies) == 0:

        return (
            0.0,
            len(audio) / sample_rate
        )

    # Adaptive threshold
    threshold = (
        np.percentile(
            energies,
            20
        ) * 2.5
    )

    # Minimum threshold
    threshold = max(
        threshold,
        0.005
    )

    # Find active speech frames
    active = energies > threshold

    active_indices = np.where(
        active
    )[0]

    # If no speech detected
    if len(active_indices) == 0:

        return (
            0.0,
            len(audio) / sample_rate
        )

    # First active frame
    start_frame = active_indices[0]

    # Last active frame
    end_frame = active_indices[-1]

    # Convert frames to seconds
    start_time = (
        start_frame * hop_size
        / sample_rate
    )

    end_time = (
        end_frame * hop_size
        + frame_size
    ) / sample_rate

    return (
        start_time,
        end_time
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 60)
print("              WORD VERIFICATION")
print("=" * 60)

print(
    f"Target word: {TARGET_WORD}"
)

print(
    f"Audio file:  {AUDIO_PATH}"
)


# ============================================================
# 1. LOAD AUDIO
# ============================================================

print("\nLoading audio...")

audio, sample_rate = sf.read(
    AUDIO_PATH
)

duration = (
    len(audio) / sample_rate
)

print(
    f"Sample rate: {sample_rate} Hz"
)

print(
    f"Duration:    {duration:.2f} seconds"
)


# ============================================================
# 2. DETECT SPEECH REGION
# ============================================================

print(
    "\nDetecting speech region..."
)

speech_start, speech_end = (
    find_speech_region(
        audio,
        sample_rate
    )
)

print(
    f"Speech start: {speech_start:.2f} sec"
)

print(
    f"Speech end:   {speech_end:.2f} sec"
)


# ============================================================
# 3. CROP SPEECH
# ============================================================

start_sample = int(
    speech_start * sample_rate
)

end_sample = int(
    speech_end * sample_rate
)

speech_audio = audio[
    start_sample:end_sample
]


# ============================================================
# 4. SAVE TEMPORARY SPEECH AUDIO
# ============================================================

sf.write(
    TEMP_AUDIO,
    speech_audio,
    sample_rate
)

print(
    f"\nSpeech audio saved: {TEMP_AUDIO}"
)


# ============================================================
# 5. LOAD WHISPER
# ============================================================

print(
    "\nLoading Whisper model..."
)

model = whisper.load_model(
    "base"
)


# ============================================================
# 6. TRANSCRIBE SPEECH ONLY
# ============================================================

print(
    "\nTranscribing speech region..."
)

result = model.transcribe(
    TEMP_AUDIO,

    language="en",

    temperature=0,

    condition_on_previous_text=False,

    fp16=False
)

recognized_text = (
    result["text"].strip()
)


# ============================================================
# 7. NORMALIZE TEXT
# ============================================================

target_normalized = (
    normalize_text(
        TARGET_WORD
    )
)

recognized_normalized = (
    normalize_text(
        recognized_text
    )
)


# ============================================================
# 8. DISPLAY RECOGNITION
# ============================================================

print(
    "\n" + "-" * 60
)

print("Target:")
print(TARGET_WORD)

print(
    "\nWhisper recognized:"
)

print(
    recognized_text
)

print(
    "\nNormalized target:"
)

print(
    target_normalized
)

print(
    "\nNormalized recognition:"
)

print(
    recognized_normalized
)

print(
    "-" * 60
)


# ============================================================
# 9. EXACT WORD CHECK
# ============================================================

recognized_words = (
    recognized_normalized.split()
)

exact_match = (
    target_normalized
    in recognized_words
)


if exact_match:

    print(
        "\nEXACT WORD CHECK: PASS"
    )

else:

    print(
        "\nEXACT WORD CHECK: REVIEW"
    )


# ============================================================
# 10. TEXT SIMILARITY
# ============================================================

score = similarity_score(
    target_normalized,
    recognized_normalized
)

print(
    f"\nText similarity: "
    f"{score * 100:.2f}%"
)


# ============================================================
# 11. INTERPRETATION
# ============================================================

print(
    "\nInterpretation:"
)

if exact_match:

    print(
        "Whisper recognized "
        "the target word."
    )

elif score >= 0.70:

    print(
        "Recognition is somewhat "
        "similar, but the target "
        "was not recognized exactly."
    )

else:

    print(
        "Recognition is substantially "
        "different from the target."
    )


# ============================================================
# 12. IMPORTANT NOTE
# ============================================================

print(
    "\nNOTE:"
)

print(
    "Whisper recognition is used "
    "only as a word-recognition signal."
)

print(
    "It is NOT used as pronunciation accuracy."
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "COMPLETE"
)

print(
    "=" * 60
)