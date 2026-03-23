"""
NumPy Basics for TTS
====================
This module covers essential NumPy operations needed for audio processing and deep learning.

Key Concepts:
- Array creation and manipulation
- Broadcasting
- Linear algebra operations
- Audio-relevant operations (FFT, convolutions)
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 1. Array Creation and Basic Operations
# =============================================================================

def create_audio_buffer(duration: float, sample_rate: int = 22050) -> np.ndarray:
    """
    Create an empty audio buffer.

    Args:
        duration: Duration in seconds
        sample_rate: Samples per second (22050 is common for TTS)

    Returns:
        Zero-filled numpy array representing audio samples
    """
    num_samples = int(duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = 22050,
    amplitude: float = 0.5
) -> np.ndarray:
    """
    Generate a pure sine wave - fundamental building block of audio.

    This demonstrates:
    - np.arange for sample indices
    - Broadcasting in the sine calculation
    - dtype handling for audio

    Args:
        frequency: Frequency in Hz (e.g., 440 for A4 note)
        duration: Duration in seconds
        sample_rate: Samples per second
        amplitude: Peak amplitude (0-1)

    Returns:
        Audio samples as float32 array
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32)


def generate_complex_tone(
    frequencies: list,
    amplitudes: list,
    duration: float,
    sample_rate: int = 22050
) -> np.ndarray:
    """
    Generate a complex tone from multiple frequencies (harmonics).

    Real speech contains multiple harmonics - this simulates that.

    Args:
        frequencies: List of frequencies in Hz
        amplitudes: Corresponding amplitudes for each frequency
        duration: Duration in seconds
        sample_rate: Samples per second

    Returns:
        Combined audio signal
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    signal = np.zeros_like(t)

    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Normalize to prevent clipping
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = signal / max_amp * 0.9

    return signal.astype(np.float32)


# =============================================================================
# 2. Array Manipulation for Audio
# =============================================================================

def pad_audio(
    audio: np.ndarray,
    target_length: int,
    mode: str = 'constant',
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad audio to a target length - common in batch processing.

    Args:
        audio: Input audio array
        target_length: Desired length
        mode: Padding mode ('constant', 'reflect', 'edge')
        pad_value: Value for constant padding

    Returns:
        Padded audio array
    """
    current_length = len(audio)

    if current_length >= target_length:
        return audio[:target_length]

    padding_needed = target_length - current_length

    if mode == 'constant':
        return np.pad(audio, (0, padding_needed), mode='constant', constant_values=pad_value)
    else:
        return np.pad(audio, (0, padding_needed), mode=mode)


def segment_audio(
    audio: np.ndarray,
    segment_length: int,
    hop_length: int
) -> np.ndarray:
    """
    Segment audio into overlapping frames - used for spectrogram computation.

    Args:
        audio: Input audio array
        segment_length: Length of each segment (frame)
        hop_length: Hop between segments (overlap = segment_length - hop_length)

    Returns:
        2D array of shape (num_frames, segment_length)
    """
    # Calculate number of frames
    num_frames = 1 + (len(audio) - segment_length) // hop_length

    # Use stride tricks for efficient frame extraction
    frames = np.zeros((num_frames, segment_length), dtype=audio.dtype)

    for i in range(num_frames):
        start = i * hop_length
        frames[i] = audio[start:start + segment_length]

    return frames


def apply_window(frames: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """
    Apply a window function to frames - reduces spectral leakage.

    Args:
        frames: 2D array of audio frames
        window_type: Type of window ('hann', 'hamming', 'blackman')

    Returns:
        Windowed frames
    """
    frame_length = frames.shape[1]

    if window_type == 'hann':
        window = np.hanning(frame_length)
    elif window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'blackman':
        window = np.blackman(frame_length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    # Broadcasting: window (frame_length,) * frames (num_frames, frame_length)
    return frames * window


# =============================================================================
# 3. Fourier Transform Basics
# =============================================================================

def compute_fft(audio: np.ndarray, n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fast Fourier Transform of audio.

    The FFT converts time-domain signal to frequency-domain.
    This is fundamental to understanding spectrograms.

    Args:
        audio: Input audio signal
        n_fft: FFT size (power of 2 for efficiency)

    Returns:
        Tuple of (frequencies, magnitudes)
    """
    # Compute FFT
    fft_result = np.fft.rfft(audio, n=n_fft)

    # Get magnitude spectrum
    magnitudes = np.abs(fft_result)

    # Get frequency bins (assuming 22050 Hz sample rate)
    frequencies = np.fft.rfftfreq(n_fft, d=1/22050)

    return frequencies, magnitudes


def compute_power_spectrum(audio: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    """
    Compute power spectrum (magnitude squared).

    Power spectrum is often used in speech processing.

    Args:
        audio: Input audio signal
        n_fft: FFT size

    Returns:
        Power spectrum
    """
    fft_result = np.fft.rfft(audio, n=n_fft)
    return np.abs(fft_result) ** 2


def compute_stft(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.

    STFT is the foundation of spectrograms - it shows how frequencies
    change over time.

    Args:
        audio: Input audio signal
        n_fft: FFT size
        hop_length: Hop between frames
        win_length: Window length (defaults to n_fft)

    Returns:
        Complex STFT matrix of shape (n_fft//2 + 1, num_frames)
    """
    if win_length is None:
        win_length = n_fft

    # Pad audio
    audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')

    # Segment into frames
    frames = segment_audio(audio_padded, win_length, hop_length)

    # Apply window
    frames = apply_window(frames, 'hann')

    # Pad frames to n_fft if needed
    if win_length < n_fft:
        pad_amount = n_fft - win_length
        frames = np.pad(frames, ((0, 0), (0, pad_amount)), mode='constant')

    # Compute FFT for each frame
    stft = np.fft.rfft(frames, n=n_fft, axis=1)

    return stft.T  # Transpose to (freq_bins, time_frames)


# =============================================================================
# 4. Linear Algebra Operations
# =============================================================================

def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to a target dB level.

    Args:
        audio: Input audio
        target_db: Target RMS level in dB

    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return audio

    # Calculate current dB
    current_db = 20 * np.log10(rms)

    # Calculate gain needed
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    return audio * gain_linear


def compute_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> np.ndarray:
    """
    Create mel filterbank matrix - converts linear frequency to mel scale.

    The mel scale approximates human perception of pitch.

    Args:
        n_fft: FFT size
        n_mels: Number of mel bands
        sample_rate: Audio sample rate
        f_min: Minimum frequency
        f_max: Maximum frequency (defaults to sample_rate / 2)

    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    if f_max is None:
        f_max = sample_rate / 2

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Create mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bin indices
    freq_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq))

    for i in range(n_mels):
        left = freq_bins[i]
        center = freq_bins[i + 1]
        right = freq_bins[i + 2]

        # Rising slope
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def apply_mel_filterbank(
    spectrogram: np.ndarray,
    mel_filterbank: np.ndarray
) -> np.ndarray:
    """
    Apply mel filterbank to a spectrogram.

    This is matrix multiplication - the core of converting
    linear spectrograms to mel spectrograms.

    Args:
        spectrogram: Linear spectrogram (freq_bins, time_frames)
        mel_filterbank: Mel filterbank (n_mels, freq_bins)

    Returns:
        Mel spectrogram (n_mels, time_frames)
    """
    return np.dot(mel_filterbank, spectrogram)


# =============================================================================
# 5. Practice Exercises
# =============================================================================

def exercise_1_array_creation():
    """Exercise: Create and manipulate audio arrays."""
    print("Exercise 1: Array Creation")
    print("-" * 40)

    # Create a 1-second audio buffer at 22050 Hz
    buffer = create_audio_buffer(1.0, 22050)
    print(f"Buffer shape: {buffer.shape}")
    print(f"Buffer dtype: {buffer.dtype}")
    print(f"Expected samples: {22050}")

    # Generate a 440 Hz sine wave
    sine = generate_sine_wave(440, 0.5)
    print(f"\nSine wave shape: {sine.shape}")
    print(f"Max amplitude: {np.max(sine):.3f}")
    print(f"Min amplitude: {np.min(sine):.3f}")


def exercise_2_spectral_analysis():
    """Exercise: Compute and analyze spectrograms."""
    print("\nExercise 2: Spectral Analysis")
    print("-" * 40)

    # Generate a complex tone
    tone = generate_complex_tone(
        frequencies=[220, 440, 880],
        amplitudes=[1.0, 0.5, 0.25],
        duration=1.0
    )

    # Compute FFT
    freqs, mags = compute_fft(tone)

    # Find peaks (fundamental frequency and harmonics)
    peak_indices = np.argsort(mags)[-10:]
    peak_freqs = freqs[peak_indices]
    print(f"Peak frequencies: {sorted(peak_freqs)[:5]}")

    # Compute STFT
    stft = compute_stft(tone)
    print(f"STFT shape: {stft.shape}")
    print(f"Frequency bins: {stft.shape[0]}")
    print(f"Time frames: {stft.shape[1]}")


def exercise_3_mel_spectrogram():
    """Exercise: Create mel spectrogram from scratch."""
    print("\nExercise 3: Mel Spectrogram")
    print("-" * 40)

    # Generate audio
    audio = generate_sine_wave(440, 1.0)

    # Compute STFT
    stft = compute_stft(audio)
    magnitude = np.abs(stft)

    # Create mel filterbank
    mel_fb = compute_mel_filterbank(
        n_fft=2048,
        n_mels=80,
        sample_rate=22050
    )

    # Apply mel filterbank
    mel_spec = apply_mel_filterbank(magnitude, mel_fb)

    print(f"Linear spectrogram shape: {magnitude.shape}")
    print(f"Mel filterbank shape: {mel_fb.shape}")
    print(f"Mel spectrogram shape: {mel_spec.shape}")

    # Convert to log scale (dB)
    mel_spec_db = 20 * np.log10(np.maximum(mel_spec, 1e-10))
    print(f"Mel spectrogram dB range: [{mel_spec_db.min():.1f}, {mel_spec_db.max():.1f}]")


if __name__ == "__main__":
    exercise_1_array_creation()
    exercise_2_spectral_analysis()
    exercise_3_mel_spectrogram()
