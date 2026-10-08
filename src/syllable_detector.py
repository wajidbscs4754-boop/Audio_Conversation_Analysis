import numpy as np
import torchaudio
from scipy.signal import find_peaks


def load_audio(audio_path):

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    audio = waveform.numpy().flatten()

    return audio, sample_rate


def calculate_energy(audio, sample_rate):

    frame_size = int(0.025 * sample_rate)
    hop_size = int(0.010 * sample_rate)

    energy = []

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

        energy.append(rms)

    return np.array(energy), hop_size


def smooth_signal(signal, window_size=9):

    kernel = (
        np.ones(window_size)
        / window_size
    )

    return np.convolve(
        signal,
        kernel,
        mode="same"
    )


def detect_syllables(audio_path):

    audio, sample_rate = load_audio(
        audio_path
    )

    energy, hop_size = calculate_energy(
        audio,
        sample_rate
    )

    # Normalize energy
    if np.max(energy) > 0:

        energy = (
            energy / np.max(energy)
        )

    # Smooth energy
    smooth_energy = smooth_signal(
        energy,
        window_size=9
    )

    # Adaptive threshold
    noise_level = np.percentile(
        smooth_energy,
        20
    )

    threshold = max(
        noise_level * 2.5,
        0.08
    )

    # Detect peaks
    peaks, _ = find_peaks(

        smooth_energy,

        distance=int(
            0.18 / 0.010
        ),

        prominence=0.04,

        height=threshold
    )

    # Convert peak positions to seconds
    peak_times = (
        peaks
        * hop_size
        / sample_rate
    )

    # Merge close peaks
    merged_peaks = []

    minimum_gap = 0.15

    for peak in peak_times:

        if not merged_peaks:

            merged_peaks.append(peak)

            continue

        if (
            peak - merged_peaks[-1]
            >= minimum_gap
        ):

            merged_peaks.append(peak)

    return np.array(merged_peaks)