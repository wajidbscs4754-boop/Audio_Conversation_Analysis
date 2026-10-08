import numpy as np
import torchaudio

AUDIO_PATH = "data/computer_test.wav"


def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    audio = waveform.numpy().flatten()

    return audio, sample_rate


def detect_speech_region(audio, sample_rate):
    frame_duration = 0.025
    hop_duration = 0.010

    frame_size = int(frame_duration * sample_rate)
    hop_size = int(hop_duration * sample_rate)

    energies = []
    times = []

    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):
        frame = audio[start:start + frame_size]

        rms = np.sqrt(np.mean(frame ** 2))

        energies.append(rms)
        times.append(start / sample_rate)

    energies = np.array(energies)
    times = np.array(times)

    noise_level = np.percentile(energies, 20)
    threshold = max(noise_level * 2.0, 0.005)

    speech_frames = energies > threshold
    speech_indices = np.where(speech_frames)[0]

    if len(speech_indices) == 0:
        return None

    first_index = speech_indices[0]
    last_index = speech_indices[-1]

    start_time = times[first_index]
    end_time = times[last_index] + frame_duration

    return start_time, end_time


def calculate_loudness(audio, sample_rate):
    frame_duration = 0.04
    hop_duration = 0.01

    frame_size = int(frame_duration * sample_rate)
    hop_size = int(hop_duration * sample_rate)

    loudness_values = []

    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):
        frame = audio[start:start + frame_size]

        rms = np.sqrt(np.mean(frame ** 2))

        if rms > 0:
            loudness_db = 20 * np.log10(rms)
            loudness_values.append(loudness_db)

    if len(loudness_values) == 0:
        return None

    return np.array(loudness_values)


def analyze_loudness(audio_path):

    print("Loading audio...")

    audio, sample_rate = load_audio(audio_path)

    speech_region = detect_speech_region(
        audio,
        sample_rate
    )

    if speech_region is None:
        print("No speech detected.")
        return

    start_time, end_time = speech_region

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    speech_audio = audio[start_sample:end_sample]

    loudness = calculate_loudness(
        speech_audio,
        sample_rate
    )

    print("\n========================================")
    print("        LOUDNESS ANALYSIS")
    print("========================================")

    print(f"Speech start:       {start_time:.2f} sec")
    print(f"Speech end:         {end_time:.2f} sec")

    if loudness is None:
        print("Could not calculate loudness.")
        return

    average_loudness = np.mean(loudness)
    minimum_loudness = np.min(loudness)
    maximum_loudness = np.max(loudness)
    loudness_variation = np.std(loudness)

    print(f"Average loudness:   {average_loudness:.2f} dB")
    print(f"Minimum loudness:   {minimum_loudness:.2f} dB")
    print(f"Maximum loudness:   {maximum_loudness:.2f} dB")
    print(f"Loudness variation: {loudness_variation:.2f} dB")

    print("========================================")


if __name__ == "__main__":
    analyze_loudness(AUDIO_PATH)