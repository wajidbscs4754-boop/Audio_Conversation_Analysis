import torch
import torchaudio

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from speech_region import load_audio, detect_speech_region


AUDIO_PATH = "data/computer_test.wav"
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"


print("Loading phoneme model...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

print("Phoneme model loaded.")


print("\nLoading audio...")

audio, sample_rate = load_audio(AUDIO_PATH)

print(f"Original sample rate: {sample_rate} Hz")
print(f"Original duration: {len(audio) / sample_rate:.2f} seconds")


# ---------------------------------------------------------
# Detect speech region
# ---------------------------------------------------------

print("\nDetecting speech region...")

speech_region = detect_speech_region(
    audio,
    sample_rate
)

if speech_region is None:
    print("No speech detected.")
    exit()


speech_start = speech_region["start_time"]
speech_end = speech_region["end_time"]

print(f"Speech start: {speech_start:.2f} sec")
print(f"Speech end:   {speech_end:.2f} sec")


# ---------------------------------------------------------
# Crop only speech
# ---------------------------------------------------------

start_sample = int(speech_start * sample_rate)
end_sample = int(speech_end * sample_rate)

speech_audio = audio[start_sample:end_sample]


print(
    f"Speech duration: "
    f"{len(speech_audio) / sample_rate:.2f} seconds"
)


# ---------------------------------------------------------
# Convert to tensor
# ---------------------------------------------------------

waveform = torch.tensor(
    speech_audio,
    dtype=torch.float32
)

if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)


# ---------------------------------------------------------
# Resample to 16 kHz
# ---------------------------------------------------------

if sample_rate != 16000:

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=16000
    )

    waveform = resampler(waveform)

    sample_rate = 16000


audio_16k = waveform.squeeze(0).numpy()


print(f"Model sample rate: {sample_rate} Hz")
print(
    f"Model audio duration: "
    f"{len(audio_16k) / sample_rate:.2f} seconds"
)


# ---------------------------------------------------------
# Phoneme recognition
# ---------------------------------------------------------

print("\nRunning phoneme recognition...")


inputs = processor(
    audio_16k,
    sampling_rate=sample_rate,
    return_tensors="pt"
)


with torch.no_grad():

    logits = model(
        inputs.input_values
    ).logits


predicted_ids = torch.argmax(
    logits,
    dim=-1
)


phonemes = processor.batch_decode(
    predicted_ids
)[0]


# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

print("\n========================================")
print("       SPEECH-ONLY PHONEME RESULT")
print("========================================")

print("Recognized phonemes:")
print(phonemes)

print("========================================")