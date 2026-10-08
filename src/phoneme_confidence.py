import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# ============================================================
# Configuration
# ============================================================

AUDIO_PATH = "data/computer_test.wav"

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

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


# ============================================================
# Load Audio
# ============================================================

def load_audio(audio_path):

    waveform, sample_rate = torchaudio.load(
        audio_path
    )

    # Stereo -> Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    audio = waveform.numpy().flatten()

    return audio, sample_rate


# ============================================================
# Calculate Frame RMS Energy
# Same logic as speech_region.py
# ============================================================

def calculate_frame_energy(
    audio,
    sample_rate,
    frame_duration=0.025,
    hop_duration=0.010
):

    frame_size = int(
        frame_duration * sample_rate
    )

    hop_size = int(
        hop_duration * sample_rate
    )

    energies = []
    times = []

    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):

        frame = audio[
            start:start + frame_size
        ]

        rms = np.sqrt(
            np.mean(frame ** 2)
        )

        energies.append(rms)

        times.append(
            start / sample_rate
        )

    return (
        np.array(energies),
        np.array(times)
    )


# ============================================================
# Detect Speech Region
# EXACT SAME LOGIC AS speech_region.py
# ============================================================

def detect_speech_region(
    audio,
    sample_rate
):

    energy, times = calculate_frame_energy(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    noise_level = np.percentile(
        energy,
        20
    )

    threshold = max(
        noise_level * 2.0,
        0.005
    )

    speech_frames = (
        energy > threshold
    )

    # --------------------------------------------------------
    # Find first and last speech frame
    # --------------------------------------------------------

    speech_indices = np.where(
        speech_frames
    )[0]

    if len(speech_indices) == 0:

        return None

    first_index = speech_indices[0]

    last_index = speech_indices[-1]

    start_time = times[
        first_index
    ]

    end_time = (
        times[last_index] + 0.025
    )

    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "threshold": threshold
    }


# ============================================================
# Load Wav2Vec2 Model
# ============================================================

def load_model():

    print("\nLoading phoneme model...")

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


# ============================================================
# Extract Speech Audio
# ============================================================

def extract_speech(
    audio,
    sample_rate,
    speech_region
):

    start_sample = int(
        speech_region["start_time"] * sample_rate
    )

    end_sample = int(
        speech_region["end_time"] * sample_rate
    )

    speech_audio = audio[
        start_sample:end_sample
    ]

    return speech_audio


# ============================================================
# Resample Audio
# Wav2Vec2 requires 16000 Hz
# ============================================================

def resample_for_model(
    speech_audio,
    sample_rate
):

    target_sample_rate = 16000

    if sample_rate == target_sample_rate:

        return speech_audio, sample_rate

    print(
        f"\nResampling audio: "
        f"{sample_rate} Hz -> {target_sample_rate} Hz"
    )

    speech_tensor = torch.tensor(
        speech_audio,
        dtype=torch.float32
    ).unsqueeze(0)

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=target_sample_rate
    )

    resampled_audio = resampler(
        speech_tensor
    )

    resampled_audio = (
        resampled_audio
        .squeeze(0)
        .numpy()
    )

    return (
        resampled_audio,
        target_sample_rate
    )


# ============================================================
# Model Prediction
# ============================================================

def predict_phonemes(
    speech_audio,
    sample_rate,
    processor,
    model
):

    # --------------------------------------------------------
    # Convert 48000 Hz -> 16000 Hz
    # --------------------------------------------------------

    speech_audio, sample_rate = resample_for_model(
        speech_audio,
        sample_rate
    )

    print(
        f"Model input sample rate: "
        f"{sample_rate} Hz"
    )

    print(
        f"Model input duration: "
        f"{len(speech_audio) / sample_rate:.2f} sec"
    )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    inputs = processor(
        speech_audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_values=inputs.input_values
        )

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    predicted_ids = torch.argmax(
        probabilities,
        dim=-1
    )[0]

    predicted_tokens = (
        processor.tokenizer.convert_ids_to_tokens(
            predicted_ids.tolist()
        )
    )

    return (
        predicted_tokens,
        probabilities[0]
    )


# ============================================================
# Clean Repeated CTC Tokens
# ============================================================

def clean_tokens(tokens):

    cleaned = []

    previous = None

    for token in tokens:

        # Ignore padding
        if token == "<pad>":

            previous = token

            continue

        # Remove consecutive duplicate tokens
        if token == previous:

            continue

        cleaned.append(token)

        previous = token

    return cleaned


# ============================================================
# Expected Phoneme Evidence
# ============================================================

def calculate_phoneme_evidence(
    expected_phonemes,
    probabilities,
    processor
):

    vocab = processor.tokenizer.get_vocab()

    results = []

    for phoneme in expected_phonemes:

        if phoneme not in vocab:

            results.append({
                "phoneme": phoneme,
                "confidence": 0.0,
                "frame": None
            })

            continue

        token_id = vocab[
            phoneme
        ]

        phoneme_probs = probabilities[
            :,
            token_id
        ]

        max_probability = torch.max(
            phoneme_probs
        ).item()

        max_frame = torch.argmax(
            phoneme_probs
        ).item()

        results.append({
            "phoneme": phoneme,
            "confidence": max_probability,
            "frame": max_frame
        })

    return results


# ============================================================
# Main Analysis
# ============================================================

def analyze():

    print("\n========================================")
    print("       PHONEME CONFIDENCE ANALYSIS")
    print("========================================")

    print(
        f"Target word: {TARGET_WORD}"
    )

    print(
        "Expected phonemes:",
        " ".join(EXPECTED_PHONEMES)
    )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print("\nLoading audio...")

    audio, sample_rate = load_audio(
        AUDIO_PATH
    )

    total_duration = (
        len(audio) / sample_rate
    )

    print(
        f"Total audio duration: "
        f"{total_duration:.2f} sec"
    )

    print(
        f"Original sample rate: "
        f"{sample_rate} Hz"
    )

    # --------------------------------------------------------
    # Detect speech region
    # --------------------------------------------------------

    speech_region = detect_speech_region(
        audio,
        sample_rate
    )

    if speech_region is None:

        print(
            "\nNo speech detected."
        )

        return

    print(
        f"Speech starts at: "
        f"{speech_region['start_time']:.2f} sec"
    )

    print(
        f"Speech ends at: "
        f"{speech_region['end_time']:.2f} sec"
    )

    print(
        f"Speech duration: "
        f"{speech_region['duration']:.2f} sec"
    )

    print(
        f"Detection threshold: "
        f"{speech_region['threshold']:.4f}"
    )

    # --------------------------------------------------------
    # Extract speech
    # --------------------------------------------------------

    speech_audio = extract_speech(
        audio,
        sample_rate,
        speech_region
    )

    print(
        f"\nSpeech samples: "
        f"{len(speech_audio)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    processor, model = load_model()

    # --------------------------------------------------------
    # Predict phonemes
    # --------------------------------------------------------

    print(
        "\nRunning phoneme model..."
    )

    predicted_tokens, probabilities = predict_phonemes(
        speech_audio,
        sample_rate,
        processor,
        model
    )

    # --------------------------------------------------------
    # Clean CTC output
    # --------------------------------------------------------

    cleaned_tokens = clean_tokens(
        predicted_tokens
    )

    print(
        "\nRecognized phonemes:"
    )

    print(
        " ".join(cleaned_tokens)
    )

    # --------------------------------------------------------
    # Expected phoneme evidence
    # --------------------------------------------------------

    evidence = calculate_phoneme_evidence(
        EXPECTED_PHONEMES,
        probabilities,
        processor
    )

    print(
        "\n========================================"
    )

    print(
        "EXPECTED PHONEME EVIDENCE"
    )

    print(
        "========================================"
    )

    for item in evidence:

        print(
            f"{item['phoneme']}: "
            f"{item['confidence']:.3f} "
            f"frame {item['frame']}"
        )

    print(
        "========================================"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    analyze()