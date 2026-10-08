
import os
import whisper
from pyannote.audio import Pipeline


# =====================================================
# FILE PATHS
# =====================================================

AUDIO_PATH = "data/conversation.wav"

TRANSCRIPT_PATH = "data/transcript.txt"
TIMESTAMP_PATH = "data/transcript_timestamps.txt"
SPEAKER_TRANSCRIPT_PATH = "data/speaker_transcript.txt"


# =====================================================
# TIME FORMAT
# =====================================================

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    return f"{minutes:02d}:{secs:02d}"


# =====================================================
# FIND SPEAKER AT TIME
# =====================================================

def get_speaker_at_time(time, diarization):

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):

        if turn.start <= time <= turn.end:
            return speaker

    return "UNKNOWN"


# =====================================================
# ASSIGN WORDS TO SPEAKERS
# =====================================================

def combine_words_with_speakers(words, diarization):

    speaker_words = []

    for word in words:

        start = word["start"]
        end = word["end"]

        midpoint = (start + end) / 2

        speaker = get_speaker_at_time(
            midpoint,
            diarization
        )

        speaker_words.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "word": word["word"].strip()
        })

    return speaker_words


# =====================================================
# GROUP WORDS INTO SPEAKER TURNS
# =====================================================

def create_speaker_segments(speaker_words):

    segments = []

    current = None

    for item in speaker_words:

        if not item["word"]:
            continue

        if current is None:

            current = {
                "start": item["start"],
                "end": item["end"],
                "speaker": item["speaker"],
                "text": item["word"]
            }

            continue

        if item["speaker"] == current["speaker"]:

            current["end"] = item["end"]

            current["text"] += (
                " " + item["word"]
            )

        else:

            segments.append(current)

            current = {
                "start": item["start"],
                "end": item["end"],
                "speaker": item["speaker"],
                "text": item["word"]
            }

    if current is not None:
        segments.append(current)

    return segments


# =====================================================
# MAIN
# =====================================================

def main():

    if not os.path.exists(AUDIO_PATH):

        print(
            f"Audio file not found: {AUDIO_PATH}"
        )

        return

    # =================================================
    # 1. WHISPER
    # =================================================

    print("\n==============================")
    print("Loading Whisper model")
    print("==============================")

    whisper_model = whisper.load_model("base")

    print("\nTranscribing audio...")

    result = whisper_model.transcribe(
        AUDIO_PATH,
        word_timestamps=True
    )

    # =================================================
    # 2. SAVE NORMAL TRANSCRIPT
    # =================================================

    transcript = result["text"].strip()

    with open(
        TRANSCRIPT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(transcript)

    print(
        f"Transcript saved to: "
        f"{TRANSCRIPT_PATH}"
    )

    # =================================================
    # 3. SAVE TIMESTAMPED TRANSCRIPT
    # =================================================

    timestamp_lines = []

    for segment in result["segments"]:

        start = segment["start"]
        end = segment["end"]

        text = segment["text"].strip()

        timestamp_lines.append(
            f"[{format_time(start)} - "
            f"{format_time(end)}] {text}"
        )

    timestamped_transcript = "\n".join(
        timestamp_lines
    )

    with open(
        TIMESTAMP_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(timestamped_transcript)

    print(
        f"Timestamped transcript saved to: "
        f"{TIMESTAMP_PATH}"
    )

    # =================================================
    # 4. SPEAKER DIARIZATION
    # =================================================

    print("\n==============================")
    print("Loading speaker diarization model")
    print("==============================")

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )

    print("\nDetecting speakers...")

    output = diarization_pipeline(
        AUDIO_PATH,
        num_speakers=2
    )

    diarization = output.speaker_diarization

    # =================================================
    # 5. GET WHISPER WORDS
    # =================================================

    words = []

    for segment in result["segments"]:

        if "words" not in segment:
            continue

        for word in segment["words"]:

            words.append(word)

    # =================================================
    # 6. ASSIGN SPEAKERS
    # =================================================

    speaker_words = combine_words_with_speakers(
        words,
        diarization
    )

    # =================================================
    # 7. CREATE SPEAKER TURNS
    # =================================================

    speaker_segments = create_speaker_segments(
        speaker_words
    )

    # =================================================
    # 8. DISPLAY SPEAKER TRANSCRIPT
    # =================================================

    print("\n==============================")
    print("Speaker-labelled transcript")
    print("==============================\n")

    output_lines = []

    for segment in speaker_segments:

        line = (
            f"[{format_time(segment['start'])} - "
            f"{format_time(segment['end'])}] "
            f"{segment['speaker']}: "
            f"{segment['text']}"
        )

        print(line)

        output_lines.append(line)

    # =================================================
    # 9. SAVE SPEAKER TRANSCRIPT
    # =================================================

    with open(
        SPEAKER_TRANSCRIPT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(output_lines)
        )

    print(
        f"\nSpeaker transcript saved to: "
        f"{SPEAKER_TRANSCRIPT_PATH}"
    )

    print("\nSpeaker processing completed.")


if __name__ == "__main__":
    main()

