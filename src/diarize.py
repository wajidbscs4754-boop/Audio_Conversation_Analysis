import os
from pyannote.audio import Pipeline

AUDIO_PATH = "data/conversation.wav"


def diarize_audio(audio_path):

    print("Loading speaker diarization model...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )

    print("Analyzing speakers...")

    output = pipeline(audio_path, num_speakers=2)

    # pyannote.audio 4.x
    diarization = output.speaker_diarization

    print("\nSpeaker Turns:\n")

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):
        print(
            f"{turn.start:.2f}s - "
            f"{turn.end:.2f}s : "
            f"{speaker}"
        )


if __name__ == "__main__":

    if not os.path.exists(AUDIO_PATH):

        print(
            f"Audio file not found: {AUDIO_PATH}"
        )

    else:

        diarize_audio(AUDIO_PATH)