import subprocess
from transformers import Wav2Vec2Processor


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

TARGET_WORD = "computer"


# ============================================================
# Get eSpeak Phonemes
# ============================================================

def get_espeak_phonemes(word):

    result = subprocess.run(
        [
            "espeak-ng",
            "-q",
            "--ipa=3",
            word
        ],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:

        print("eSpeak error:")
        print(result.stderr)

        return None

    return result.stdout.strip()


# ============================================================
# Main
# ============================================================

def main():

    print("\n========================================")
    print("       PHONEME VOCABULARY TEST")
    print("========================================")

    # --------------------------------------------------------
    # eSpeak pronunciation
    # --------------------------------------------------------

    print(
        f"\nTarget word: {TARGET_WORD}"
    )

    espeak_phonemes = get_espeak_phonemes(
        TARGET_WORD
    )

    print(
        "\neSpeak pronunciation:"
    )

    print(
        espeak_phonemes
    )

    # --------------------------------------------------------
    # Load Wav2Vec2 processor
    # --------------------------------------------------------

    print(
        "\nLoading Wav2Vec2 processor..."
    )

    processor = (
        Wav2Vec2Processor.from_pretrained(
            MODEL_NAME
        )
    )

    vocab = (
        processor.tokenizer.get_vocab()
    )

    print(
        f"Vocabulary size: {len(vocab)}"
    )

    # --------------------------------------------------------
    # Check individual characters/symbols
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "CHECKING eSpeak PHONEMES"
    )

    print(
        "========================================"
    )

    if espeak_phonemes:

        unique_symbols = []

        for symbol in espeak_phonemes:

            if symbol not in unique_symbols:

                unique_symbols.append(symbol)

        for symbol in unique_symbols:

            if symbol in vocab:

                token_id = vocab[symbol]

                print(
                    f"[FOUND]  {symbol} "
                    f"-> token id {token_id}"
                )

            else:

                print(
                    f"[MISSING] {symbol}"
                )

    # --------------------------------------------------------
    # Show relevant vocabulary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "RELEVANT VOCABULARY"
    )

    print(
        "========================================"
    )

    relevant_tokens = []

    for token in vocab:

        if any(
            character in token
            for character in [
                "k",
                "ə",
                "m",
                "p",
                "j",
                "u",
                "ɹ",
                "ɚ",
                "t",
                "ʌ",
                "ɐ"
            ]
        ):

            relevant_tokens.append(token)

    for token in sorted(
        relevant_tokens
    )[:100]:

        print(token)

    print(
        "\n========================================"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()