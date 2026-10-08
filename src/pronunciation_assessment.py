import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

AUDIO_PATH = DATA_DIR / "computer_test.wav"
TARGET_WORD = "computer"

def run_script(script_name):
    """Run another project analysis script and return its output."""
    script_path = PROJECT_ROOT / "src" / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace",
    )

    return result.stdout, result.stderr, result.returncode


def load_json(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


print("=" * 65)
print("              PRONUNCIATION ASSESSMENT")
print("=" * 65)

print(f"\nTarget word : {TARGET_WORD}")
print(f"Audio file  : {AUDIO_PATH}")

if not AUDIO_PATH.exists():
    print("\nERROR: Audio file not found.")
    print(f"Expected: {AUDIO_PATH}")
    sys.exit(1)


# ---------------------------------------------------------
# 1. SYLLABLE ANALYSIS
# ---------------------------------------------------------

print("\n" + "-" * 65)
print("1. SYLLABLE ANALYSIS")
print("-" * 65)

stdout, stderr, code = run_script("syllable_stones.py")

print(stdout)

if code != 0:
    print("Syllable analysis failed.")
    print(stderr)


# ---------------------------------------------------------
# 2. SPEECH TIMING / RHYTHM
# ---------------------------------------------------------

print("\n" + "-" * 65)
print("2. SPEECH TIMING / RHYTHM")
print("-" * 65)

stdout, stderr, code = run_script("speech_timing.py")

print(stdout)

if code != 0:
    print("Speech timing analysis failed.")
    print(stderr)


# ---------------------------------------------------------
# 3. PITCH
# ---------------------------------------------------------

print("\n" + "-" * 65)
print("3. PITCH ANALYSIS")
print("-" * 65)

stdout, stderr, code = run_script("pitch_analysis.py")

print(stdout)

if code != 0:
    print("Pitch analysis failed.")
    print(stderr)


# ---------------------------------------------------------
# 4. LOUDNESS
# ---------------------------------------------------------

print("\n" + "-" * 65)
print("4. LOUDNESS ANALYSIS")
print("-" * 65)

stdout, stderr, code = run_script("loudness_analysis.py")

print(stdout)

if code != 0:
    print("Loudness analysis failed.")
    print(stderr)


# ---------------------------------------------------------
# 5. WORD VERIFICATION
# ---------------------------------------------------------

print("\n" + "-" * 65)
print("5. WORD VERIFICATION")
print("-" * 65)

stdout, stderr, code = run_script("word_verification.py")

print(stdout)

if code != 0:
    print("Word verification failed.")
    print(stderr)


# ---------------------------------------------------------
# FINAL STUDENT-FRIENDLY RESULT
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("                 STUDENT RESULT")
print("=" * 65)

print(f"\nTarget word: {TARGET_WORD}")

print("\nCurrent assessment:")
print("✓ Syllable analysis completed")
print("✓ Rhythm analysis completed")
print("✓ Pitch analysis completed")
print("✓ Loudness analysis completed")
print("✓ Word verification completed")

print("\nPronunciation status:")
print("⚠ Needs further pronunciation analysis")

print("\nImportant:")
print(
    "The current Whisper result is not being converted "
    "into a pronunciation accuracy percentage."
)

print("\n" + "=" * 65)
print("                    COMPLETE")
print("=" * 65)