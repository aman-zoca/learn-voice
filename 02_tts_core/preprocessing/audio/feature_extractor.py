"""
Audio Feature Extractor Module
==============================
Extracts mel spectrograms and other features from audio for TTS.
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class FeatureConfig:
    """Configuration for audio feature extraction."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    center: bool = True
    pad_mode: str = 'reflect'
    power: float = 1.0  # 1.0 for energy, 2.0 for power
    norm: Optional[str] = 'slaney'
    mel_scale: str = 'htk'  # 'htk' or 'slaney'
    ref_level_db: float = 20.0
    min_level_db: float = -100.0
    symmetric_norm: bool = True
    max_abs_value: float = 4.0
    clip_val: float = 1e-5


class MelSpectrogramExtractor:
    """
    Extract mel spectrograms from audio.

    This is the primary feature representation used in TTS models
    like Tacotron2 and VITS.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._mel_basis = None

        # Initialize torch mel spectrogram transform if available
        if TORCH_AVAILABLE:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                f_min=self.config.f_min,
                f_max=self.config.f_max,
                n_mels=self.config.n_mels,
                power=self.config.power,
                center=self.config.center,
                pad_mode=self.config.pad_mode,
                mel_scale=self.config.mel_scale
            )
        else:
            self._mel_transform = None

    @property
    def mel_basis(self) -> np.ndarray:
        """Get or create mel filterbank."""
        if self._mel_basis is None:
            if LIBROSA_AVAILABLE:
                self._mel_basis = librosa.filters.mel(
                    sr=self.config.sample_rate,
                    n_fft=self.config.n_fft,
                    n_mels=self.config.n_mels,
                    fmin=self.config.f_min,
                    fmax=self.config.f_max,
                    norm=self.config.norm
                )
            else:
                self._mel_basis = self._create_mel_basis()

        return self._mel_basis

    def _create_mel_basis(self) -> np.ndarray:
        """Create mel filterbank manually."""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(self.config.f_min)
        mel_max = hz_to_mel(self.config.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.config.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        freq_bins = np.floor(
            (self.config.n_fft + 1) * hz_points / self.config.sample_rate
        ).astype(int)

        n_freq = self.config.n_fft // 2 + 1
        filterbank = np.zeros((self.config.n_mels, n_freq))

        for i in range(self.config.n_mels):
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
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window='hann',
                center=self.config.center,
                pad_mode=self.config.pad_mode
            )

        # Manual STFT
        if self.config.center:
            audio = np.pad(
                audio,
                (self.config.n_fft // 2, self.config.n_fft // 2),
                mode=self.config.pad_mode
            )

        num_frames = 1 + (len(audio) - self.config.n_fft) // self.config.hop_length
        window = np.hanning(self.config.win_length)

        stft_matrix = np.zeros(
            (self.config.n_fft // 2 + 1, num_frames),
            dtype=np.complex64
        )

        for i in range(num_frames):
            start = i * self.config.hop_length
            frame = audio[start:start + self.config.win_length]

            windowed = np.zeros(self.config.n_fft)
            windowed[:len(frame)] = frame * window

            stft_matrix[:, i] = np.fft.rfft(windowed)

        return stft_matrix

    def linear_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute linear magnitude spectrogram.

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
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute mel spectrogram.

        Args:
            audio: Input audio array
            normalize: Apply normalization

        Returns:
            Mel spectrogram (n_mels, time_frames)
        """
        # Compute magnitude spectrogram
        magnitude = self.linear_spectrogram(audio)

        # Apply mel filterbank
        mel_spec = np.dot(self.mel_basis, magnitude)

        # Ensure minimum value
        mel_spec = np.maximum(mel_spec, self.config.clip_val)

        # Convert to log scale
        mel_spec = np.log(mel_spec)

        if normalize:
            mel_spec = self.normalize_mel(mel_spec)

        return mel_spec.astype(np.float32)

    def mel_spectrogram_torch(
        self,
        audio: 'torch.Tensor',
        normalize: bool = True
    ) -> 'torch.Tensor':
        """
        Compute mel spectrogram using PyTorch (faster for batches).

        Args:
            audio: Input audio tensor (batch, samples) or (samples,)
            normalize: Apply normalization

        Returns:
            Mel spectrogram tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mel_spec = self._mel_transform(audio)

        # Convert to log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=self.config.clip_val))

        if normalize:
            mel_spec = self.normalize_mel_torch(mel_spec)

        return mel_spec

    def normalize_mel(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Normalize mel spectrogram to range [-max_abs_value, max_abs_value].

        Args:
            mel_spec: Log mel spectrogram

        Returns:
            Normalized mel spectrogram
        """
        # Scale to [0, 1]
        mel_spec = (mel_spec - self.config.min_level_db) / (
            -self.config.min_level_db
        )

        if self.config.symmetric_norm:
            # Scale to [-max_abs_value, max_abs_value]
            mel_spec = self.config.max_abs_value * (2 * mel_spec - 1)
        else:
            # Scale to [0, max_abs_value]
            mel_spec = self.config.max_abs_value * mel_spec

        return mel_spec

    def normalize_mel_torch(self, mel_spec: 'torch.Tensor') -> 'torch.Tensor':
        """Normalize mel spectrogram (torch version)."""
        mel_spec = (mel_spec - self.config.min_level_db) / (
            -self.config.min_level_db
        )

        if self.config.symmetric_norm:
            mel_spec = self.config.max_abs_value * (2 * mel_spec - 1)
        else:
            mel_spec = self.config.max_abs_value * mel_spec

        return mel_spec

    def denormalize_mel(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Denormalize mel spectrogram.

        Args:
            mel_spec: Normalized mel spectrogram

        Returns:
            Log mel spectrogram
        """
        if self.config.symmetric_norm:
            mel_spec = (mel_spec / self.config.max_abs_value + 1) / 2
        else:
            mel_spec = mel_spec / self.config.max_abs_value

        mel_spec = mel_spec * (-self.config.min_level_db) + self.config.min_level_db

        return mel_spec

    def mel_to_linear(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Convert mel spectrogram back to linear spectrogram (approximate).

        Uses pseudo-inverse of mel filterbank.

        Args:
            mel_spec: Mel spectrogram

        Returns:
            Approximate linear spectrogram
        """
        mel_basis_pinv = np.linalg.pinv(self.mel_basis)
        return np.maximum(np.dot(mel_basis_pinv, mel_spec), 0)


class GriffinLimVocoder:
    """
    Griffin-Lim algorithm for spectrogram to audio conversion.

    This is a simple baseline vocoder useful for testing before
    implementing neural vocoders.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 256,
        win_length: int = 1024,
        n_iter: int = 60
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_iter = n_iter

    def __call__(
        self,
        magnitude: np.ndarray,
        n_iter: Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct audio from magnitude spectrogram.

        Args:
            magnitude: Magnitude spectrogram (freq_bins, time_frames)
            n_iter: Number of iterations (overrides default)

        Returns:
            Reconstructed audio
        """
        n_iter = n_iter or self.n_iter

        if LIBROSA_AVAILABLE:
            return librosa.griffinlim(
                magnitude,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.n_fft
            )

        # Manual implementation
        angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        complex_spec = magnitude * angles

        for _ in range(n_iter):
            audio = self._istft(complex_spec)
            complex_spec = self._stft(audio)
            angles = np.exp(1j * np.angle(complex_spec))
            complex_spec = magnitude * angles

        return self._istft(complex_spec)

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT."""
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
            windowed = np.zeros(self.n_fft)
            windowed[:len(frame)] = frame * window
            stft_matrix[:, i] = np.fft.rfft(windowed)

        return stft_matrix

    def _istft(self, stft_matrix: np.ndarray) -> np.ndarray:
        """Compute inverse STFT."""
        num_frames = stft_matrix.shape[1]
        audio_length = (num_frames - 1) * self.hop_length + self.n_fft

        audio = np.zeros(audio_length)
        window = np.hanning(self.win_length)
        window_sum = np.zeros(audio_length)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = np.fft.irfft(stft_matrix[:, i])[:self.win_length]
            audio[start:start + self.win_length] += frame * window
            window_sum[start:start + self.win_length] += window ** 2

        nonzero = window_sum > 1e-10
        audio[nonzero] /= window_sum[nonzero]

        return audio[self.n_fft // 2:-self.n_fft // 2]


class PitchExtractor:
    """
    Extract pitch (F0) from audio.

    Pitch is important for prosody and voice characteristics.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 600.0
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract(
        self,
        audio: np.ndarray,
        method: str = 'pyin'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch from audio.

        Args:
            audio: Input audio
            method: Extraction method ('pyin' or 'crepe')

        Returns:
            Tuple of (f0, voiced_flag, voiced_prob)
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for pitch extraction")

        if method == 'pyin':
            f0, voiced_flag, voiced_prob = librosa.pyin(
                audio,
                fmin=self.f0_min,
                fmax=self.f0_max,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
        else:
            raise ValueError(f"Unknown pitch extraction method: {method}")

        # Replace NaN with 0
        f0 = np.nan_to_num(f0)

        return f0, voiced_flag, voiced_prob

    def interpolate(self, f0: np.ndarray, voiced_flag: np.ndarray) -> np.ndarray:
        """
        Interpolate unvoiced regions of F0.

        Args:
            f0: F0 contour
            voiced_flag: Boolean array indicating voiced frames

        Returns:
            Interpolated F0
        """
        f0_interp = f0.copy()

        # Find voiced regions
        voiced_indices = np.where(voiced_flag)[0]

        if len(voiced_indices) == 0:
            return f0_interp

        # Linear interpolation
        unvoiced_indices = np.where(~voiced_flag)[0]
        f0_interp[unvoiced_indices] = np.interp(
            unvoiced_indices,
            voiced_indices,
            f0[voiced_indices]
        )

        return f0_interp


class EnergyExtractor:
    """
    Extract energy/amplitude envelope from audio.

    Used for prosody modeling and duration prediction.
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 256
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract frame-level energy.

        Args:
            audio: Input audio

        Returns:
            Energy contour
        """
        # Pad audio
        pad_length = self.frame_length // 2
        audio_padded = np.pad(audio, (pad_length, pad_length), mode='reflect')

        # Calculate number of frames
        num_frames = 1 + (len(audio_padded) - self.frame_length) // self.hop_length

        # Extract energy per frame
        energy = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio_padded[start:start + self.frame_length]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        return energy.astype(np.float32)

    def extract_from_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Extract energy from spectrogram (sum over frequency bins).

        Args:
            spectrogram: Magnitude spectrogram (freq_bins, time_frames)

        Returns:
            Energy contour
        """
        return np.sum(spectrogram, axis=0).astype(np.float32)


# Factory function for easy instantiation
def create_feature_extractor(config: Optional[FeatureConfig] = None) -> MelSpectrogramExtractor:
    """Create a configured mel spectrogram extractor."""
    return MelSpectrogramExtractor(config)


def create_vocoder(
    n_fft: int = 2048,
    hop_length: int = 256,
    win_length: int = 1024
) -> GriffinLimVocoder:
    """Create a Griffin-Lim vocoder."""
    return GriffinLimVocoder(n_fft, hop_length, win_length)
