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


def calculate_pitch(audio, sample_rate):
    frame_duration = 0.04
    hop_duration = 0.01

    frame_size = int(frame_duration * sample_rate)
    hop_size = int(hop_duration * sample_rate)

    pitches = []

    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):
        frame = audio[start:start + frame_size]

        # Remove DC offset
        frame = frame - np.mean(frame)

        # Autocorrelation
        correlation = np.correlate(frame, frame, mode="full")
        correlation = correlation[len(correlation) // 2:]

        # Human speech pitch range
        min_frequency = 70
        max_frequency = 400

        min_lag = int(sample_rate / max_frequency)
        max_lag = int(sample_rate / min_frequency)

        if max_lag >= len(correlation):
            continue

        search_region = correlation[min_lag:max_lag]

        if len(search_region) == 0:
            continue

        peak_index = np.argmax(search_region)

        lag = peak_index + min_lag

        if lag == 0:
            continue

        pitch = sample_rate / lag

        # Ignore unreliable values
        if min_frequency <= pitch <= max_frequency:
            pitches.append(pitch)

    if len(pitches) == 0:
        return None

    return np.array(pitches)


def analyze_pitch(audio_path):

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

    pitches = calculate_pitch(
        speech_audio,
        sample_rate
    )

    print("\n========================================")
    print("          PITCH ANALYSIS")
    print("========================================")

    print(f"Speech start:       {start_time:.2f} sec")
    print(f"Speech end:         {end_time:.2f} sec")

    if pitches is None:
        print("Could not detect pitch.")
        return

    average_pitch = np.mean(pitches)
    minimum_pitch = np.min(pitches)
    maximum_pitch = np.max(pitches)

    print(f"Average pitch:      {average_pitch:.2f} Hz")
    print(f"Minimum pitch:      {minimum_pitch:.2f} Hz")
    print(f"Maximum pitch:      {maximum_pitch:.2f} Hz")

    print("========================================")


if __name__ == "__main__":
    analyze_pitch(AUDIO_PATH)