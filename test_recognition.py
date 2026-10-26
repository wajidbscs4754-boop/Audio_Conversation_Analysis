
import whisper
from pathlib import Path

audio_path = Path("data/syllable_test_new.wav").resolve()

if not audio_path.is_file():
    raise FileNotFoundError(f"Audio not found: {audio_path}")

print("Loading Whisper base model...")
model = whisper.load_model("base")

tests = [
    {"name": "Default", "options": {}},
    {"name": "No previous context", "options": {
        "condition_on_previous_text": False,
        "temperature": 0
    }},
    {"name": "Lower no-speech threshold", "options": {
        "condition_on_previous_text": False,
        "temperature": 0,
        "no_speech_threshold": 0.3
    }},
]

for test in tests:
    print(f"\n--- {test['name']} ---")
    result = model.transcribe(
        str(audio_path),
        language="en",
        fp16=False,
        **test["options"]
    )
    print("Recognized:", repr(result["text"].strip()))
    print("No-speech probability:",
          round(result["segments"][0]["no_speech_prob"], 3)
          if result["segments"] else "No segments")
