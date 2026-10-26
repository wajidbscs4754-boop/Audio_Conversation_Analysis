
import torch
import torchaudio
import numpy as np

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer import phonemize
from phonemizer.separator import Separator

from speech_region import detect_speech_region


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_PATH = "data/computer_test.wav"
TARGET_WORD = "computer"

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
TARGET_SAMPLE_RATE = 16000


# ============================================================
# EXPECTED PHONEMES
# ============================================================

def get_expected_phonemes(word):
    result = phonemize(
        word,
        language="en-us",
        backend="espeak",
        separator=Separator(
            phone=" ",
            word="|",
            syllable=""
        ),
        strip=True
    )

    phonemes = result.split()

    if not phonemes:
        raise ValueError(
            f"No phonemes generated for word: {word}"
        )

    return phonemes


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.numel() == 0:
        raise ValueError("The audio file is empty.")

    # Convert stereo/multichannel audio to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(torch.float32)

    return waveform, sample_rate


# ============================================================
# DETECT AND CROP SPEECH
# ============================================================

def get_speech_audio(audio_path):
    waveform, sample_rate = load_audio(audio_path)

    audio = waveform.squeeze(0).cpu().numpy()

    speech_region = detect_speech_region(
        audio,
        sample_rate
    )

    if speech_region is None:
        raise ValueError(
            "No speech region detected. "
            "Please record the target word again."
        )

    start_time = max(
        0.0,
        float(speech_region["start_time"])
    )

    end_time = min(
        len(audio) / sample_rate,
        float(speech_region["end_time"])
    )

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    if end_sample <= start_sample:
        raise ValueError(
            "Invalid speech region detected."
        )

    speech_waveform = waveform[
        :,
        start_sample:end_sample
    ]

    if speech_waveform.numel() == 0:
        raise ValueError("Cropped speech audio is empty.")

    # Return original sample rate as well.
    return (
        speech_waveform.squeeze(0).numpy(),
        speech_region,
        sample_rate
    )


# ============================================================
# RESAMPLE AUDIO
# ============================================================

def resample_audio(audio, original_sample_rate):
    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    if original_sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=TARGET_SAMPLE_RATE
        )

        waveform = resampler(waveform)

    result = waveform.squeeze(0).cpu().numpy()

    if result.size == 0:
        raise ValueError(
            "Audio is empty after resampling."
        )

    return result


# ============================================================
# LOAD MODEL ONCE
# ============================================================

def load_phoneme_model():
    print("Loading phoneme model...")

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


# ============================================================
# GET MODEL PROBABILITIES
# ============================================================

def get_probabilities(audio, processor, model):
    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=False
    )

    with torch.no_grad():
        logits = model(
            inputs.input_values
        ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    return probabilities.squeeze(0)


# ============================================================
# CTC ALIGNMENT
# ============================================================

def ctc_align(probabilities, target_ids, blank_id):
    if probabilities.ndim != 2:
        raise ValueError(
            "Probabilities must have shape [frames, vocabulary]."
        )

    num_frames = probabilities.shape[0]

    if num_frames == 0:
        raise ValueError("The model produced no frames.")

    if not target_ids:
        raise ValueError("No target phoneme IDs provided.")

    if num_frames < len(target_ids):
        raise ValueError(
            "Audio is too short to align all expected phonemes. "
            "Please record the word clearly and slightly slower."
        )

    log_probs = torch.log(
        probabilities.clamp_min(1e-10)
    )

    # Blank, phone, blank, phone, ..., blank.
    expanded = [blank_id]

    for phoneme_id in target_ids:
        expanded.append(phoneme_id)
        expanded.append(blank_id)

    num_states = len(expanded)

    dp = torch.full(
        (num_frames, num_states),
        -float("inf"),
        dtype=log_probs.dtype,
        device=log_probs.device
    )

    backtrack = torch.zeros(
        (num_frames, num_states),
        dtype=torch.long,
        device=log_probs.device
    )

    # First frame can start in the initial blank
    # or the first phoneme state.
    dp[0, 0] = log_probs[0, expanded[0]]
    dp[0, 1] = log_probs[0, expanded[1]]

    for t in range(1, num_frames):
        for s in range(num_states):
            current_id = expanded[s]

            best_score = dp[t - 1, s]
            best_state = s

            # Move from the immediately previous state.
            if s > 0:
                previous_score = dp[t - 1, s - 1]

                if previous_score > best_score:
                    best_score = previous_score
                    best_state = s - 1

            # Skip a blank state when CTC permits it.
            if s > 1:
                skip_allowed = (
                    current_id != blank_id
                    and current_id != expanded[s - 2]
                )

                if skip_allowed:
                    skip_score = dp[t - 1, s - 2]

                    if skip_score > best_score:
                        best_score = skip_score
                        best_state = s - 2

            dp[t, s] = (
                best_score + log_probs[t, current_id]
            )

            backtrack[t, s] = best_state

    # A valid completed alignment ends in the final
    # blank state or the final phoneme state.
    final_states = [
        num_states - 1,
        num_states - 2
    ]

    best_final_state = max(
        final_states,
        key=lambda state: dp[num_frames - 1, state].item()
    )

    final_score = dp[
        num_frames - 1,
        best_final_state
    ].item()

    if not np.isfinite(final_score):
        raise ValueError(
            "CTC alignment failed. Check the target word, "
            "phoneme vocabulary and recording."
        )

    # Backtrack to recover the state assigned to each frame.
    states = []
    current_state = best_final_state

    for t in range(num_frames - 1, -1, -1):
        states.append(current_state)

        if t > 0:
            current_state = int(
                backtrack[t, current_state].item()
            )

    states.reverse()

    return final_score, states, expanded


# ============================================================
# PHONEME SEGMENTS
# ============================================================

def get_phoneme_segments(
    states,
    expanded,
    probabilities,
    target_ids
):
    segments = []

    for index, phoneme_id in enumerate(target_ids):
        state_index = index * 2 + 1

        frames = [
            frame_index
            for frame_index, state in enumerate(states)
            if state == state_index
        ]

        if not frames:
            segments.append({
                "phoneme": expanded[state_index],
                "start_frame": None,
                "end_frame": None,
                "confidence": 0.0
            })
            continue

        frame_probs = probabilities[
            frames,
            phoneme_id
        ]

        confidence = frame_probs.mean().item()

        segments.append({
            "phoneme": expanded[state_index],
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "confidence": confidence
        })

    return segments


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("          PHONEME ALIGNMENT TEST")
    print("=" * 60)

    print(f"Audio file: {AUDIO_PATH}")
    print(f"Target word: {TARGET_WORD}")

    # Expected phonemes
    expected = get_expected_phonemes(TARGET_WORD)

    print("\nExpected phonemes:")
    print(" ".join(expected))

    # Load and crop speech
    print("\nDetecting speech region...")

    speech_audio, speech_region, original_sample_rate = (
        get_speech_audio(AUDIO_PATH)
    )

    print(
        f"Original sample rate: "
        f"{original_sample_rate} Hz"
    )

    print(
        f"Speech start: "
        f"{speech_region['start_time']:.2f}s"
    )

    print(
        f"Speech end: "
        f"{speech_region['end_time']:.2f}s"
    )

    print(
        f"Detected duration: "
        f"{speech_region['duration']:.2f}s"
    )

    # Resample using the actual input sample rate
    speech_audio_16k = resample_audio(
        speech_audio,
        original_sample_rate
    )

    # Load model
    processor, model = load_phoneme_model()

    # Acoustic probabilities
    probabilities = get_probabilities(
        speech_audio_16k,
        processor,
        model
    )

    print(f"\nModel frames: {probabilities.shape[0]}")

    # Match phonemes to tokenizer vocabulary
    vocabulary = processor.tokenizer.get_vocab()

    target_ids = []

    print("\nPhoneme IDs:")

    for phoneme in expected:
        phoneme_id = vocabulary.get(phoneme)

        if phoneme_id is None:
            raise ValueError(
                f"Phoneme '{phoneme}' is not present in "
                "the model tokenizer vocabulary. "
                "The phonemizer output and model vocabulary "
                "may use different phoneme conventions."
            )

        if phoneme_id == processor.tokenizer.pad_token_id:
            raise ValueError(
                f"Phoneme '{phoneme}' unexpectedly maps "
                "to the CTC blank token."
            )

        target_ids.append(phoneme_id)

        print(f"{phoneme:<10} -> {phoneme_id}")

    blank_id = processor.tokenizer.pad_token_id

    if blank_id is None:
        raise ValueError(
            "The tokenizer does not define a CTC blank token."
        )

    print(f"\nBlank ID: {blank_id}")

    # Align the expected phonemes to acoustic frames
    print("\nRunning CTC alignment...")

    score, states, expanded = ctc_align(
        probabilities,
        target_ids,
        blank_id
    )

    print(f"CTC alignment score (log space): {score:.2f}")

    segments = get_phoneme_segments(
        states,
        expanded,
        probabilities,
        target_ids
    )

    # The cropped audio begins at this absolute time.
    speech_start_time = float(
        speech_region["start_time"]
    )

    # Approximate model frame step.
    # Model output frames are not guaranteed to be
    # perfectly uniformly aligned to the waveform duration.
    audio_duration = len(speech_audio_16k) / TARGET_SAMPLE_RATE
    frame_duration = audio_duration / probabilities.shape[0]

    print("\nPhoneme alignment:")
    print("-" * 60)

    for segment in segments:
        phoneme = segment["phoneme"]
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]
        confidence = segment["confidence"] * 100

        if start_frame is None:
            print(f"{phoneme:<10} NOT ALIGNED")
            continue

        start_time = (
            speech_start_time
            + start_frame * frame_duration
        )

        end_time = (
            speech_start_time
            + (end_frame + 1) * frame_duration
        )

        print(
            f"{phoneme:<10} "
            f"{start_time:.2f}s - {end_time:.2f}s  "
            f"acoustic evidence: {confidence:.2f}%"
        )

    print("-" * 60)
    print(
        "\nImportant: Acoustic evidence is not a calibrated "
        "pronunciation accuracy score."
    )

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
