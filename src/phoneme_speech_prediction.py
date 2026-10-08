import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


AUDIO_PATH = "data/computer_test.wav"
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

TARGET_WORD = "computer"
TARGET_PHONEMES = ["k", "ɹ", "iː", "eɪ", "ɾ", "ɚ"]

# Speech region already identified from our previous analysis
SPEECH_START = 0.85
SPEECH_END = 3.80

BLANK_ID = 0


def collapse_ctc(ids):
    """
    CTC decoding:
    - Remove consecutive duplicate tokens
    - Remove blank tokens
    """

    collapsed = []
    previous = None

    for token_id in ids:

        token_id = int(token_id)

        # Remove consecutive duplicates
        if token_id == previous:
            continue

        previous = token_id

        # Remove CTC blank
        if token_id == BLANK_ID:
            continue

        collapsed.append(token_id)

    return collapsed


print("=" * 55)
print("        SPEECH-ONLY PHONEME PREDICTION")
print("=" * 55)

print(f"Target word: {TARGET_WORD}")

print("\nExpected phonemes:")
print(" ".join(TARGET_PHONEMES))


# --------------------------------------------------
# 1. Load audio
# --------------------------------------------------

print("\nLoading audio...")

waveform, sample_rate = torchaudio.load(AUDIO_PATH)

print(f"Original sample rate: {sample_rate} Hz")


# --------------------------------------------------
# 2. Crop speech region
# --------------------------------------------------

print("\nUsing speech region...")

print(f"Speech start: {SPEECH_START:.2f}s")
print(f"Speech end:   {SPEECH_END:.2f}s")
print(
    f"Speech duration: "
    f"{SPEECH_END - SPEECH_START:.2f}s"
)


start_sample = int(SPEECH_START * sample_rate)
end_sample = int(SPEECH_END * sample_rate)

speech_audio = waveform[:, start_sample:end_sample]


# --------------------------------------------------
# 3. Convert stereo to mono
# --------------------------------------------------

if speech_audio.shape[0] > 1:

    speech_audio = speech_audio.mean(
        dim=0,
        keepdim=True
    )


# --------------------------------------------------
# 4. Resample to 16 kHz
# --------------------------------------------------

if sample_rate != 16000:

    print("\nResampling audio to 16 kHz...")

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=16000
    )

    speech_audio = resampler(speech_audio)

    sample_rate = 16000


speech_audio = speech_audio.squeeze(0)

print(f"Model sample rate: {sample_rate} Hz")
print(f"Speech samples: {speech_audio.shape[0]}")


# --------------------------------------------------
# 5. Load model
# --------------------------------------------------

print("\nLoading phoneme model...")

processor = Wav2Vec2Processor.from_pretrained(
    MODEL_NAME
)

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME
)

model.eval()


# --------------------------------------------------
# 6. Run model
# --------------------------------------------------

print("\nRunning model prediction...")

inputs = processor(
    speech_audio.numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():

    outputs = model(**inputs)


logits = outputs.logits


# --------------------------------------------------
# 7. Get best token for every frame
# --------------------------------------------------

predicted_ids = torch.argmax(
    logits,
    dim=-1
)[0]

print(f"Model frames: {len(predicted_ids)}")


# --------------------------------------------------
# 8. Raw prediction
# --------------------------------------------------

print("\nRaw CTC prediction IDs:")

print(predicted_ids.tolist())


# --------------------------------------------------
# 9. Collapse CTC
# --------------------------------------------------

collapsed_ids = collapse_ctc(
    predicted_ids.tolist()
)


# --------------------------------------------------
# 10. Convert IDs to phoneme names
# --------------------------------------------------

tokens = []

for token_id in collapsed_ids:

    token = processor.tokenizer.convert_ids_to_tokens(
        token_id
    )

    tokens.append(token)


# --------------------------------------------------
# 11. Display prediction
# --------------------------------------------------

print("\n" + "=" * 55)

print("Speech-only model prediction:")

if tokens:
    print(" ".join(tokens))
else:
    print("(no phonemes detected)")


print("\nCollapsed prediction:")

if tokens:
    print(" ".join(tokens))
else:
    print("(empty)")


print("\nExpected:")
print(" ".join(TARGET_PHONEMES))

print("=" * 55)


# --------------------------------------------------
# 12. Basic statistics
# --------------------------------------------------

print("\nNumber of predicted phonemes:", len(tokens))

print(
    "Number of expected phonemes:",
    len(TARGET_PHONEMES)
)


print("\nCOMPLETE")
