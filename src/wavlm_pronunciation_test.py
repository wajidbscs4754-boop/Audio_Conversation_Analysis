from pathlib import Path

from wavlm_pipeline import PronunciationAssessorV2


# =====================================================
# TEST SETTINGS
# =====================================================

AUDIO_PATH = Path(
    "data/computer_reference.wav"
)

TARGET_WORD = "computer"


# =====================================================
# CHECK AUDIO
# =====================================================

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_PATH}"
    )


print("\n==============================")
print("WavLM Pronunciation Test")
print("==============================")

print(f"Audio: {AUDIO_PATH}")
print(f"Target word: {TARGET_WORD}")


# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading WavLM pronunciation model...")

assessor = PronunciationAssessorV2.from_pretrained()

print("Model loaded successfully.")


# =====================================================
# RUN PRONUNCIATION ASSESSMENT
# =====================================================

print("\nAssessing pronunciation...")

result = assessor.assess(
    str(AUDIO_PATH),
    TARGET_WORD
)


# =====================================================
# DISPLAY RESULT
# =====================================================

print("\n==============================")
print("PRONUNCIATION RESULT")
print("==============================")

print(result)