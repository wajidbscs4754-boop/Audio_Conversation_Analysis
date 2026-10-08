import os
import re
import json
import whisper
from difflib import SequenceMatcher

AUDIO_PATH = "data/computer_test.wav"
TARGET_WORD = "computer"

print("=" * 50)
print("       PRONUNCIATION WORD CHECK")
print("=" * 50)

# --------------------------------------------------
# 1. Check audio file
# --------------------------------------------------

if not os.path.exists(AUDIO_PATH):
    print(f"Audio file not found: {AUDIO_PATH}")
    raise SystemExit

print(f"Target word: {TARGET_WORD}")
print(f"Audio file:  {AUDIO_PATH}")

# --------------------------------------------------
# 2. Load Whisper
# --------------------------------------------------

print("\nLoading speech recognition model...")

model = whisper.load_model("base")

# --------------------------------------------------
# 3. Transcribe
# --------------------------------------------------

print("Analyzing student's speech...")

result = model.transcribe(
    AUDIO_PATH,
    language="en",
    fp16=False
)

recognized_text = result["text"].strip().lower()

print(f"\nRecognized speech: {recognized_text}")

# --------------------------------------------------
# 4. Clean text
# --------------------------------------------------

clean_target = re.sub(r"[^a-z]", "", TARGET_WORD.lower())
clean_recognized = re.sub(r"[^a-z]", "", recognized_text)

# --------------------------------------------------
# 5. Similarity
# --------------------------------------------------

similarity = SequenceMatcher(
    None,
    clean_target,
    clean_recognized
).ratio()

similarity_percentage = similarity * 100

print(f"\nText similarity: {similarity_percentage:.2f}%")

# --------------------------------------------------
# 6. Basic pronunciation evidence
# --------------------------------------------------

if clean_recognized == clean_target:
    word_status = "WORD MATCH"
elif similarity >= 0.75:
    word_status = "CLOSE WORD MATCH"
else:
    word_status = "WORD NOT CONFIRMED"

print(f"Word status: {word_status}")

# --------------------------------------------------
# 7. Save result
# --------------------------------------------------

assessment = {
    "target_word": TARGET_WORD,
    "recognized_text": recognized_text,
    "text_similarity": round(similarity_percentage, 2),
    "word_status": word_status
}

with open(
    "data/pronunciation_similarity.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(assessment, f, indent=4)

print("\nResult saved:")
print("data/pronunciation_similarity.json")

print("=" * 50)