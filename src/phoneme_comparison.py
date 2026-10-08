from phoneme_model_test import MODEL_NAME
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from phonemizer import phonemize


AUDIO_PATH = "data/computer_test.wav"
TARGET_WORD = "computer"


def get_expected_phonemes(word):
    result = phonemize(
        word,
        language="en-us",
        backend="espeak",
        strip=True
    )

    return result.replace(" ", "")


def recognize_phonemes(audio_path):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        waveform = resampler(waveform)

    audio = waveform.squeeze(0).numpy()

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    recognized = processor.batch_decode(predicted_ids)[0]

    return recognized.replace(" ", "")


def levenshtein_distance(expected, recognized):

    rows = len(expected) + 1
    cols = len(recognized) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):

        for j in range(1, cols):

            if expected[i - 1] == recognized[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


def calculate_similarity(expected, recognized):

    distance = levenshtein_distance(
        expected,
        recognized
    )

    max_length = max(
        len(expected),
        len(recognized)
    )

    if max_length == 0:
        return 0

    score = (1 - distance / max_length) * 100

    return max(0, score)


print("Loading...")

expected = get_expected_phonemes(TARGET_WORD)

recognized = recognize_phonemes(AUDIO_PATH)

score = calculate_similarity(
    expected,
    recognized
)


print("\n========================================")
print("       PHONEME COMPARISON")
print("========================================")

print(f"Target word:        {TARGET_WORD}")

print(f"Expected phonemes:  {expected}")

print(f"Recognized phonemes: {recognized}")

print(f"\nPhoneme similarity: {score:.2f}%")

print("========================================")