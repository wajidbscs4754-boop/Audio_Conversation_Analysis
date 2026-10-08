import torch
import torchaudio

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer import phonemize
from phonemizer.separator import Separator

from speech_region import detect_speech_region


AUDIO_PATH = "data/computer_test.wav"
TARGET_WORD = "computer"

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"


# ============================================================
# Expected phonemes
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

    return result.split()


# ============================================================
# Load full audio
# ============================================================

def load_audio(audio_path):

    waveform, sample_rate = torchaudio.load(
        audio_path
    )

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    return waveform, sample_rate


# ============================================================
# Crop speech region
# ============================================================

def get_speech_audio(audio_path):

    waveform, sample_rate = load_audio(
        audio_path
    )

    audio = waveform.squeeze(0).numpy()

    speech_region = detect_speech_region(
        audio,
        sample_rate
    )

    if speech_region is None:
        raise ValueError(
            "No speech region detected."
        )

    start_sample = int(
        speech_region["start_time"]
        * sample_rate
    )

    end_sample = int(
        speech_region["end_time"]
        * sample_rate
    )

    speech_waveform = waveform[
        :,
        start_sample:end_sample
    ]

    return speech_waveform.squeeze(0).numpy(), speech_region


# ============================================================
# Model probabilities
# ============================================================

def get_probabilities(audio):

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        logits = model(
            inputs.input_values
        ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    return processor, probabilities.squeeze(0)


# ============================================================
# Proper CTC alignment
# ============================================================

def ctc_align(
    probabilities,
    target_ids,
    blank_id
):

    log_probs = torch.log(
        probabilities + 1e-10
    )

    num_frames = log_probs.shape[0]

    expanded = []

    for phoneme_id in target_ids:

        expanded.append(blank_id)
        expanded.append(phoneme_id)

    expanded.append(blank_id)

    num_states = len(expanded)

    dp = torch.full(
        (num_frames, num_states),
        -float("inf")
    )

    backtrack = torch.zeros(
        (num_frames, num_states),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # First frame
    # --------------------------------------------------------

    dp[0, 0] = log_probs[
        0,
        blank_id
    ]

    if num_states > 1:

        dp[0, 1] = log_probs[
            0,
            expanded[1]
        ]

    # --------------------------------------------------------
    # Dynamic programming
    # --------------------------------------------------------

    for t in range(1, num_frames):

        for s in range(num_states):

            current_id = expanded[s]

            best_score = dp[
                t - 1,
                s
            ]

            best_state = s

            # Previous state
            if s > 0:

                previous_score = dp[
                    t - 1,
                    s - 1
                ]

                if previous_score > best_score:

                    best_score = previous_score
                    best_state = s - 1

            # Skip state
            if s > 1:

                skip_allowed = (
                    current_id != blank_id
                    and current_id != expanded[s - 2]
                )

                if skip_allowed:

                    skip_score = dp[
                        t - 1,
                        s - 2
                    ]

                    if skip_score > best_score:

                        best_score = skip_score
                        best_state = s - 2

            dp[
                t,
                s
            ] = (
                best_score
                + log_probs[
                    t,
                    current_id
                ]
            )

            backtrack[
                t,
                s
            ] = best_state

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    final_states = [
        num_states - 1,
        num_states - 2
    ]

    best_final_state = max(
        final_states,
        key=lambda s: dp[
            num_frames - 1,
            s
        ]
    )

    final_score = dp[
        num_frames - 1,
        best_final_state
    ].item()

    # --------------------------------------------------------
    # Backtrack
    # --------------------------------------------------------

    states = []

    current_state = best_final_state

    for t in range(
        num_frames - 1,
        -1,
        -1
    ):

        states.append(
            current_state
        )

        current_state = backtrack[
            t,
            current_state
        ].item()

    states.reverse()

    return final_score, states, expanded


# ============================================================
# Get phoneme segments
# ============================================================

def get_phoneme_segments(
    states,
    expanded,
    probabilities,
    target_ids
):

    segments = []

    for index, phoneme_id in enumerate(
        target_ids
    ):

        state_index = (
            index * 2 + 1
        )

        frames = [
            frame
            for frame, state in enumerate(states)
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

        confidence = (
            frame_probs.mean().item()
        )

        segments.append({
            "phoneme": expanded[state_index],
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "confidence": confidence
        })

    return segments


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 55)
    print("        PRONUNCIATION ASSESSMENT")
    print("=" * 55)

    print(
        f"Target word: {TARGET_WORD}"
    )

    # --------------------------------------------------------
    # Expected phonemes
    # --------------------------------------------------------

    expected = get_expected_phonemes(
        TARGET_WORD
    )

    print("\nExpected phonemes:")
    print(" ".join(expected))

    # --------------------------------------------------------
    # Speech region
    # --------------------------------------------------------

    print("\nDetecting speech region...")

    speech_audio, speech_region = get_speech_audio(
        AUDIO_PATH
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
        f"Speech duration: "
        f"{speech_region['duration']:.2f}s"
    )

    # --------------------------------------------------------
    # Resample speech audio
    # --------------------------------------------------------

    waveform = torch.tensor(
        speech_audio
    ).unsqueeze(0)

    resampler = torchaudio.transforms.Resample(
        orig_freq=48000,
        new_freq=16000
    )

    waveform = resampler(
        waveform
    )

    speech_audio_16k = waveform.squeeze(0).numpy()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading phoneme model...")

    processor, probabilities = get_probabilities(
        speech_audio_16k
    )

    print(
        f"Model frames: "
        f"{probabilities.shape[0]}"
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocabulary = (
        processor.tokenizer.get_vocab()
    )

    target_ids = []

    print("\nPhoneme IDs:")

    for phoneme in expected:

        phoneme_id = vocabulary.get(
            phoneme
        )

        if phoneme_id is None:

            raise ValueError(
                f"{phoneme} not found "
                f"in vocabulary."
            )

        target_ids.append(
            phoneme_id
        )

        print(
            f"{phoneme:<5} -> {phoneme_id}"
        )

    blank_id = (
        processor.tokenizer.pad_token_id
    )

    print(
        f"\nBlank ID: {blank_id}"
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    print(
        "\nRunning CTC alignment..."
    )

    score, states, expanded = ctc_align(
        probabilities,
        target_ids,
        blank_id
    )

    print(
        f"CTC alignment score: "
        f"{score:.2f}"
    )

    # --------------------------------------------------------
    # Segments
    # --------------------------------------------------------

    segments = get_phoneme_segments(
        states,
        expanded,
        probabilities,
        target_ids
    )

    print("\nPhoneme alignment:")
    print("-" * 55)

    frame_duration = (
        speech_region["duration"]
        / probabilities.shape[0]
    )

    for segment in segments:

        phoneme = segment["phoneme"]

        start_frame = (
            segment["start_frame"]
        )

        end_frame = (
            segment["end_frame"]
        )

        confidence = (
            segment["confidence"]
            * 100
        )

        if start_frame is None:

            print(
                f"{phoneme:<5} "
                f"NOT ALIGNED"
            )

            continue

        start_time = (
            speech_region["start_time"]
            + start_frame
            * frame_duration
        )

        end_time = (
            speech_region["start_time"]
            + (end_frame + 1)
            * frame_duration
        )

        print(
            f"{phoneme:<5} "
            f"{start_time:.2f}s - "
            f"{end_time:.2f}s   "
            f"confidence: "
            f"{confidence:.2f}%"
        )

    print("-" * 55)

    print(
        "\nNote: The confidence values are "
        "acoustic model evidence, not final "
        "pronunciation accuracy."
    )

    print()
    print("=" * 55)
    print("             COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()