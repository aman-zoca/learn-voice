"""
Audio Processing Fundamentals for TTS
=====================================
This module covers essential audio processing concepts using librosa and torchaudio.

Key Concepts:
- Audio loading and saving
- Sample rates and resampling
- Spectrograms and mel spectrograms
- Audio normalization and preprocessing
"""

import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path

# Conditional imports for flexibility
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# =============================================================================
# 1. Audio Loading and Saving
# =============================================================================

def load_audio(
    filepath: Union[str, Path],
    target_sr: int = 22050,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample.

    Args:
        filepath: Path to audio file
        target_sr: Target sample rate (22050 is common for TTS)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    filepath = Path(filepath)

    if LIBROSA_AVAILABLE:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=mono)
        return audio, sr

    elif SOUNDFILE_AVAILABLE:
        audio, sr = sf.read(filepath)
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            # Simple resampling using numpy
            ratio = target_sr / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
        return audio.astype(np.float32), target_sr

    else:
        raise ImportError("Please install librosa or soundfile")


def save_audio(
    filepath: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int = 22050
):
    """
    Save audio array to file.

    Args:
        filepath: Output path
        audio: Audio array
        sample_rate: Sample rate
    """
    filepath = Path(filepath)

    if SOUNDFILE_AVAILABLE:
        sf.write(filepath, audio, sample_rate)
    elif LIBROSA_AVAILABLE:
        import soundfile as sf
        sf.write(filepath, audio, sample_rate)
    else:
        raise ImportError("Please install soundfile")


def load_audio_torch(
    filepath: Union[str, Path],
    target_sr: int = 22050
) -> Tuple['torch.Tensor', int]:
    """
    Load audio using torchaudio (faster for training).

    Args:
        filepath: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Please install torch and torchaudio")

    waveform, sr = torchaudio.load(filepath)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


# =============================================================================
# 2. Sample Rate and Bit Depth
# =============================================================================

def explain_sample_rate():
    """
    Explain sample rate concepts with examples.

    Sample rate (Hz): Number of samples per second
    - 8000 Hz: Telephone quality
    - 16000 Hz: Wideband speech
    - 22050 Hz: Common for TTS (good quality, reasonable size)
    - 44100 Hz: CD quality
    - 48000 Hz: Professional audio
    """
    sample_rates = {
        8000: "Telephone quality (narrowband)",
        16000: "Wideband speech (ASR common)",
        22050: "TTS standard (half of CD)",
        44100: "CD quality",
        48000: "Professional audio/video"
    }

    print("Sample Rate Reference:")
    print("-" * 50)
    for sr, description in sample_rates.items():
        # Nyquist frequency = sr / 2
        nyquist = sr / 2
        print(f"{sr:6d} Hz: {description}")
        print(f"         Max frequency: {nyquist:.0f} Hz")
        print()


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to a different sample rate.

    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if LIBROSA_AVAILABLE:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    else:
        # Simple linear interpolation (not as good as librosa)
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# =============================================================================
# 3. Spectrogram Computation
# =============================================================================

class SpectrogramExtractor:
    """
    Extract spectrograms from audio.

    This class provides both numpy/librosa and torch implementations.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2

        # Create mel filterbank
        if LIBROSA_AVAILABLE:
            self.mel_basis = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=f_min,
                fmax=self.f_max
            )
        else:
            self.mel_basis = self._create_mel_basis()

    def _create_mel_basis(self) -> np.ndarray:
        """Create mel filterbank using numpy."""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        freq_bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        n_freq = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, n_freq))

        for i in range(self.n_mels):
            left = freq_bins[i]
            center = freq_bins[i + 1]
            right = freq_bins[i + 2]

            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)

            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.

        Args:
            audio: Input audio array

        Returns:
            Complex STFT matrix (freq_bins, time_frames)
        """
        if LIBROSA_AVAILABLE:
            return librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann',
                center=True,
                pad_mode='reflect'
            )
        else:
            # Manual STFT computation
            audio_padded = np.pad(
                audio,
                (self.n_fft // 2, self.n_fft // 2),
                mode='reflect'
            )

            num_frames = 1 + (len(audio_padded) - self.n_fft) // self.hop_length
            window = np.hanning(self.win_length)

            stft_matrix = np.zeros(
                (self.n_fft // 2 + 1, num_frames),
                dtype=np.complex64
            )

            for i in range(num_frames):
                start = i * self.hop_length
                frame = audio_padded[start:start + self.win_length]

                # Apply window and zero-pad
                windowed = np.zeros(self.n_fft)
                windowed[:len(frame)] = frame * window

                # FFT
                stft_matrix[:, i] = np.fft.rfft(windowed)

            return stft_matrix

    def linear_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute linear (magnitude) spectrogram.

        Args:
            audio: Input audio

        Returns:
            Magnitude spectrogram
        """
        stft_result = self.stft(audio)
        return np.abs(stft_result)

    def mel_spectrogram(
        self,
        audio: np.ndarray,
        log_scale: bool = True,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: float = 80.0
    ) -> np.ndarray:
        """
        Compute mel spectrogram.

        This is the primary feature representation for TTS.

        Args:
            audio: Input audio
            log_scale: Convert to log scale (dB)
            ref: Reference value for dB conversion
            amin: Minimum amplitude
            top_db: Dynamic range (dB)

        Returns:
            Mel spectrogram
        """
        # Get magnitude spectrogram
        magnitude = self.linear_spectrogram(audio)

        # Apply mel filterbank
        mel_spec = np.dot(self.mel_basis, magnitude)

        if log_scale:
            # Convert to dB
            mel_spec = 20 * np.log10(np.maximum(mel_spec, amin) / ref)

            # Clip to dynamic range
            mel_spec = np.maximum(mel_spec, mel_spec.max() - top_db)

        return mel_spec

    def mel_spectrogram_torch(
        self,
        audio: 'torch.Tensor',
        log_scale: bool = True
    ) -> 'torch.Tensor':
        """
        Compute mel spectrogram using torch (for training).

        Args:
            audio: Input audio tensor (batch, samples) or (samples,)
            log_scale: Convert to log scale

        Returns:
            Mel spectrogram tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("Please install torch and torchaudio")

        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # STFT
        stft_result = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(audio.device),
            center=True,
            pad_mode='reflect',
            return_complex=True
        )

        # Magnitude
        magnitude = torch.abs(stft_result)

        # Apply mel filterbank
        mel_basis_torch = torch.from_numpy(self.mel_basis).float().to(audio.device)
        mel_spec = torch.matmul(mel_basis_torch, magnitude)

        if log_scale:
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))

        return mel_spec


# =============================================================================
# 4. Audio Normalization and Preprocessing
# =============================================================================

def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0
) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Input audio
        target_db: Target RMS level in dB

    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)

    return audio * gain_linear


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 22050,
    top_db: float = 20.0
) -> np.ndarray:
    """
    Trim leading and trailing silence.

    Args:
        audio: Input audio
        sample_rate: Sample rate
        top_db: Threshold below peak (dB)

    Returns:
        Trimmed audio
    """
    if LIBROSA_AVAILABLE:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    else:
        # Simple energy-based trimming
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        # Compute energy
        energy = np.array([
            np.sum(audio[i:i + frame_length] ** 2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])

        # Threshold
        threshold = np.max(energy) * (10 ** (-top_db / 10))

        # Find non-silent frames
        non_silent = energy > threshold
        if not np.any(non_silent):
            return audio

        start_frame = np.argmax(non_silent)
        end_frame = len(non_silent) - np.argmax(non_silent[::-1])

        start_sample = start_frame * hop_length
        end_sample = min(end_frame * hop_length + frame_length, len(audio))

        return audio[start_sample:end_sample]


def apply_preemphasis(
    audio: np.ndarray,
    coef: float = 0.97
) -> np.ndarray:
    """
    Apply pre-emphasis filter to audio.

    Pre-emphasis boosts high frequencies, compensating for
    the natural roll-off in speech.

    Args:
        audio: Input audio
        coef: Pre-emphasis coefficient (typically 0.95-0.97)

    Returns:
        Filtered audio
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def remove_preemphasis(
    audio: np.ndarray,
    coef: float = 0.97
) -> np.ndarray:
    """
    Remove pre-emphasis filter (de-emphasis).

    Args:
        audio: Pre-emphasized audio
        coef: Pre-emphasis coefficient

    Returns:
        De-emphasized audio
    """
    return np.array([
        sum(coef ** i * audio[j - i] for i in range(j + 1))
        for j in range(len(audio))
    ], dtype=np.float32)


# =============================================================================
# 5. Griffin-Lim Algorithm
# =============================================================================

def griffin_lim(
    magnitude: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 256,
    win_length: int = 1024,
    n_iter: int = 60
) -> np.ndarray:
    """
    Reconstruct audio from magnitude spectrogram using Griffin-Lim.

    This is a simple vocoder that can be used for testing before
    implementing neural vocoders.

    Args:
        magnitude: Magnitude spectrogram
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_iter: Number of iterations

    Returns:
        Reconstructed audio
    """
    if LIBROSA_AVAILABLE:
        return librosa.griffinlim(
            magnitude,
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft
        )
    else:
        # Manual Griffin-Lim implementation
        # Initialize with random phase
        angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        complex_spec = magnitude * angles

        for _ in range(n_iter):
            # Inverse STFT
            audio = _istft(complex_spec, hop_length, win_length)

            # Forward STFT
            complex_spec = _stft(audio, n_fft, hop_length, win_length)

            # Enforce magnitude constraint
            angles = np.exp(1j * np.angle(complex_spec))
            complex_spec = magnitude * angles

        return _istft(complex_spec, hop_length, win_length)


def _stft(audio, n_fft, hop_length, win_length):
    """Helper function for manual STFT."""
    audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2), mode='reflect')
    num_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    window = np.hanning(win_length)

    stft_matrix = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop_length
        frame = audio_padded[start:start + win_length]
        windowed = np.zeros(n_fft)
        windowed[:len(frame)] = frame * window
        stft_matrix[:, i] = np.fft.rfft(windowed)

    return stft_matrix


def _istft(stft_matrix, hop_length, win_length):
    """Helper function for manual inverse STFT."""
    n_fft = (stft_matrix.shape[0] - 1) * 2
    num_frames = stft_matrix.shape[1]
    audio_length = (num_frames - 1) * hop_length + n_fft

    audio = np.zeros(audio_length)
    window = np.hanning(win_length)
    window_sum = np.zeros(audio_length)

    for i in range(num_frames):
        start = i * hop_length
        frame = np.fft.irfft(stft_matrix[:, i])[:win_length]
        audio[start:start + win_length] += frame * window
        window_sum[start:start + win_length] += window ** 2

    # Normalize by window sum
    nonzero = window_sum > 1e-10
    audio[nonzero] /= window_sum[nonzero]

    return audio[n_fft // 2:-n_fft // 2]


# =============================================================================
# 6. Practice Examples
# =============================================================================

def example_spectrogram_extraction():
    """Demonstrate spectrogram extraction."""
    print("Spectrogram Extraction Example")
    print("-" * 40)

    # Generate test audio (1 second of 440 Hz sine wave)
    sr = 22050
    t = np.arange(0, 1.0, 1.0 / sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")

    # Extract features
    extractor = SpectrogramExtractor(sample_rate=sr)

    linear_spec = extractor.linear_spectrogram(audio)
    print(f"Linear spectrogram shape: {linear_spec.shape}")

    mel_spec = extractor.mel_spectrogram(audio)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"Mel spectrogram range: [{mel_spec.min():.1f}, {mel_spec.max():.1f}] dB")


def example_audio_preprocessing():
    """Demonstrate audio preprocessing pipeline."""
    print("\nAudio Preprocessing Example")
    print("-" * 40)

    # Generate test audio with some silence
    sr = 22050
    silence = np.zeros(int(0.5 * sr))
    t = np.arange(0, 1.0, 1.0 / sr)
    speech = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    print(f"Original length: {len(audio)} samples ({len(audio)/sr:.2f}s)")

    # Trim silence
    trimmed = trim_silence(audio, sr)
    print(f"Trimmed length: {len(trimmed)} samples ({len(trimmed)/sr:.2f}s)")

    # Normalize
    normalized = normalize_audio(trimmed, target_db=-20)
    rms = np.sqrt(np.mean(normalized ** 2))
    rms_db = 20 * np.log10(rms)
    print(f"Normalized RMS: {rms_db:.1f} dB")

    # Pre-emphasis
    preemphasized = apply_preemphasis(normalized)
    print(f"Pre-emphasized audio length: {len(preemphasized)}")


def example_griffin_lim():
    """Demonstrate Griffin-Lim reconstruction."""
    print("\nGriffin-Lim Reconstruction Example")
    print("-" * 40)

    # Generate test audio
    sr = 22050
    t = np.arange(0, 0.5, 1.0 / sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Extract spectrogram
    extractor = SpectrogramExtractor(sample_rate=sr)
    magnitude = extractor.linear_spectrogram(audio)

    print(f"Original audio length: {len(audio)}")
    print(f"Magnitude spectrogram shape: {magnitude.shape}")

    # Reconstruct
    reconstructed = griffin_lim(magnitude, n_iter=30)

    # Compare (they won't be identical due to phase loss)
    min_len = min(len(audio), len(reconstructed))
    correlation = np.corrcoef(audio[:min_len], reconstructed[:min_len])[0, 1]
    print(f"Reconstructed audio length: {len(reconstructed)}")
    print(f"Correlation with original: {correlation:.3f}")


if __name__ == "__main__":
    explain_sample_rate()
    example_spectrogram_extraction()
    example_audio_preprocessing()
    example_griffin_lim()
