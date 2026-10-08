from pathlib import Path
import whisper


# =============================
# FILE PATHS
# =============================

audio_path = "data/conversation.wav"
transcript_path = Path("data/transcript.txt")
timestamp_path = Path("data/transcript_timestamps.txt")


# =============================
# LOAD WHISPER
# =============================

print("Loading Whisper model...")

model = whisper.load_model("base")


# =============================
# TRANSCRIBE AUDIO
# =============================

print("Transcribing audio...")

result = model.transcribe(audio_path)


# =============================
# NORMAL TRANSCRIPT
# =============================

transcript = result["text"].strip()

transcript_path.write_text(
    transcript,
    encoding="utf-8"
)


# =============================
# TIMESTAMPED TRANSCRIPT
# =============================

timestamp_lines = []

for segment in result["segments"]:

    start = segment["start"]
    end = segment["end"]
    text = segment["text"].strip()

    start_minutes = int(start // 60)
    start_seconds = int(start % 60)

    end_minutes = int(end // 60)
    end_seconds = int(end % 60)

    timestamp_lines.append(
        f"[{start_minutes:02d}:{start_seconds:02d} - "
        f"{end_minutes:02d}:{end_seconds:02d}] {text}"
    )


timestamped_transcript = "\n".join(timestamp_lines)


timestamp_path.write_text(
    timestamped_transcript,
    encoding="utf-8"
)


# =============================
# OUTPUT
# =============================

print("\nTranscript")
print("------------------------------")
print(transcript)

print("\nTimestamped Transcript")
print("------------------------------")
print(timestamped_transcript)

print("\nFiles created:")
print(f"- {transcript_path}")
print(f"- {timestamp_path}")