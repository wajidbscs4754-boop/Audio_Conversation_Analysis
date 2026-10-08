import os
import re
import json

INPUT_FILE = "data/speaker_transcript.txt"
OUTPUT_FILE = "data/conversation_turns.json"


def parse_time(time_text):
    """
    Convert MM:SS into seconds.
    """
    minutes, seconds = time_text.split(":")
    return int(minutes) * 60 + int(seconds)


def calculate_speech_rate(text, duration):
    """
    Calculate words spoken per minute.
    """
    words = text.split()
    word_count = len(words)

    if duration <= 0:
        return 0

    words_per_minute = (word_count / duration) * 60

    return round(words_per_minute, 2)


def parse_speaker_transcript():

    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    turns = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        lines = file.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        pattern = (
            r"\[(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})\]\s*"
            r"(SPEAKER_\d+):\s*(.*)"
        )

        match = re.match(
            pattern,
            line
        )

        if not match:
            continue

        start_text = match.group(1)
        end_text = match.group(2)
        speaker = match.group(3)
        text = match.group(4)

        start = parse_time(start_text)
        end = parse_time(end_text)

        duration = round(
            end - start,
            2
        )

        word_count = len(text.split())

        speech_rate = calculate_speech_rate(
            text,
            duration
        )

        turn = {
            "speaker": speaker,
            "start": start,
            "end": end,
            "duration": duration,
            "word_count": word_count,
            "speech_rate_wpm": speech_rate,
            "text": text
        }

        turns.append(turn)

    # Save JSON
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            turns,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nConversation turns saved to:")
    print(OUTPUT_FILE)

    print(f"\nTotal turns: {len(turns)}")

    print("\nTurns:\n")

    for turn in turns:

        print(
            f"{turn['speaker']} | "
            f"{turn['start']:.2f}s - "
            f"{turn['end']:.2f}s | "
            f"Words: {turn['word_count']} | "
            f"Rate: {turn['speech_rate_wpm']} WPM | "
            f"{turn['text']}"
        )


if __name__ == "__main__":
    parse_speaker_transcript()