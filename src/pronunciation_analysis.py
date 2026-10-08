import whisper


# ============================================================
# Configuration
# ============================================================

AUDIO_PATH = "data/computer_test.wav"

TARGET_WORD = "computer"


# ============================================================
# Load Whisper Model
# ============================================================

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Whisper model loaded.")


# ============================================================
# Transcribe Audio
# ============================================================

print("\nAnalyzing pronunciation...")

result = model.transcribe(
    AUDIO_PATH,
    language="en",
    temperature=0,
    initial_prompt=TARGET_WORD,
    condition_on_previous_text=False
)


# ============================================================
# Get Recognized Text
# ============================================================

recognized_text = result["text"].strip()


print("\n========================================")
print("       PRONUNCIATION ANALYSIS")
print("========================================")

print(
    f"Target word:     {TARGET_WORD}"
)

print(
    f"Recognized text: {recognized_text}"
)


# ============================================================
# Normalize Text
# ============================================================

target = TARGET_WORD.lower().strip()

recognized = recognized_text.lower().strip()


# Remove common punctuation
recognized = recognized.replace(".", "")
recognized = recognized.replace(",", "")
recognized = recognized.replace("!", "")
recognized = recognized.replace("?", "")


# ============================================================
# Word Recognition
# ============================================================

if target == recognized:

    pronunciation_match = True

    print("\nWord Recognition:")
    print("PASS - Target word was recognized.")

else:

    pronunciation_match = False

    print("\nWord Recognition:")
    print("NEEDS REVIEW - Target word was not recognized exactly.")


# ============================================================
# Result
# ============================================================

result_data = {

    "target_word":
        TARGET_WORD,

    "recognized_text":
        recognized_text,

    "word_recognition":
        pronunciation_match
}


print("\nResult:")
print(result_data)

print("========================================")