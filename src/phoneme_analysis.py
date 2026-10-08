import numpy as np
import torch
import torchaudio

from speech_region import load_audio, detect_speech_region
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


AUDIO_PATH = "data/computer_test.wav"

TARGET_WORD = "computer"

EXPECTED_PHONEMES = [
    "k",
    "ə",
    "m",
    "p",
    "j",
    "uː",
    "ɾ",
    "ɚ"
]
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def load_speech_audio(audio_path):

    audio, sample_rate = load_audio(audio_path)

    speech_region = detect_speech_region(
        audio,
        sample_rate
    )

    if speech_region is None:
        return None, None, None

    start = int(
        speech_region["start_time"] * sample_rate
    )

    end = int(
        speech_region["end_time"] * sample_rate
    )

    speech_audio = audio[start:end]

    waveform = torch.tensor(
        speech_audio,
        dtype=torch.float32
    )

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != 16000:

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )

        waveform = resampler(waveform)
        sample_rate = 16000

    return (
        waveform.squeeze(0).numpy(),
        sample_rate,
        speech_region
    )


def phoneme_evidence(audio_16k, sample_rate):

    print("\nLoading phoneme model...")

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    print("Phoneme model loaded.")

    inputs = processor(
        audio_16k,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    with torch.no_grad():

        logits = model(
            inputs.input_values
        ).logits

    log_probs = torch.log_softmax(
        logits,
        dim=-1
    )[0]

    predicted_ids = torch.argmax(
        log_probs,
        dim=-1
    )

    predicted_phonemes = processor.batch_decode(
        predicted_ids.unsqueeze(0)
    )[0]

    return log_probs, predicted_phonemes


def get_phoneme_ids(processor):

    vocabulary = processor.tokenizer.get_vocab()

    phoneme_ids = {}

    for phoneme in EXPECTED_PHONEMES:

        if phoneme in vocabulary:

            phoneme_ids[phoneme] = vocabulary[phoneme]

        else:

            phoneme_ids[phoneme] = None

    return phoneme_ids


def analyze_phonemes(audio_path):

    print("Loading audio...")

    audio_16k, sample_rate, speech_region = (
        load_speech_audio(audio_path)
    )

    if audio_16k is None:

        print("No speech detected.")

        return None

    print(
        f"Speech region: "
        f"{speech_region['start_time']:.2f} - "
        f"{speech_region['end_time']:.2f} sec"
    )

    print(
        f"Speech duration: "
        f"{len(audio_16k) / sample_rate:.2f} sec"
    )

    print("\n========================================")
    print("       PHONEME EVIDENCE")
    print("========================================")

    print(f"Target word: {TARGET_WORD}")

    print("\nExpected phonemes:")

    for i, phoneme in enumerate(
        EXPECTED_PHONEMES,
        start=1
    ):

        print(f"  {i}. /{phoneme}/")

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME
    )

    log_probs, predicted_phonemes = (
        phoneme_evidence(
            audio_16k,
            sample_rate
        )
    )

    print("\nModel phoneme sequence:")
    print(f"  {predicted_phonemes}")

    phoneme_ids = get_phoneme_ids(
        processor
    )

    print("\nTarget phoneme evidence:")

    evidence = []

    for phoneme in EXPECTED_PHONEMES:

        phoneme_id = phoneme_ids[phoneme]

        if phoneme_id is None:

            print(
                f"  /{phoneme}/ -> "
                "not available in model vocabulary"
            )

            evidence.append({
                "phoneme": phoneme,
                "status": "UNAVAILABLE"
            })

            continue

        scores = log_probs[:, phoneme_id]

        best_score = torch.max(scores).item()

        evidence.append({
            "phoneme": phoneme,
            "best_score": best_score,
            "status": "EVIDENCE_AVAILABLE"
        })

        print(
            f"  /{phoneme}/ -> "
            f"evidence score: {best_score:.2f}"
        )

    print("\n========================================")
    print("Interpretation")
    print("========================================")

    print(
        "The model sequence is shown as diagnostic evidence."
    )

    print(
        "Target phoneme scores are also diagnostic evidence."
    )

    print(
        "These values are NOT converted into a pronunciation "
        "percentage."
    )

    print(
        "A final student pronunciation decision requires "
        "validated phoneme-level assessment."
    )

    print("========================================")

    return {
        "target_word": TARGET_WORD,
        "expected_phonemes": EXPECTED_PHONEMES,
        "model_phonemes": predicted_phonemes,
        "phoneme_evidence": evidence
    }


if __name__ == "__main__":

    analyze_phonemes(AUDIO_PATH)