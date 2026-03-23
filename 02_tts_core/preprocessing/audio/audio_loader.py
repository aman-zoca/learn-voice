"""
Audio Loader Module
===================
Handles loading, validation, and basic audio operations for TTS.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List
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

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    clip_val: float = 1e-5
    max_wav_value: float = 32768.0


class AudioLoader:
    """
    Audio loading and basic processing utilities.

    Provides consistent interface regardless of backend (librosa/torchaudio).
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def load(
        self,
        filepath: Union[str, Path],
        sample_rate: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic format detection.

        Args:
            filepath: Path to audio file
            sample_rate: Target sample rate (uses config if None)
            mono: Convert to mono

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        filepath = Path(filepath)
        target_sr = sample_rate or self.config.sample_rate

        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(filepath, sr=target_sr, mono=mono)
        elif TORCH_AVAILABLE:
            audio, sr = self._load_with_torchaudio(filepath, target_sr, mono)
        elif SOUNDFILE_AVAILABLE:
            audio, sr = self._load_with_soundfile(filepath, target_sr, mono)
        else:
            raise ImportError(
                "No audio backend available. Install librosa, torchaudio, or soundfile."
            )

        return audio.astype(np.float32), sr

    def _load_with_torchaudio(
        self,
        filepath: Path,
        target_sr: int,
        mono: bool
    ) -> Tuple[np.ndarray, int]:
        """Load using torchaudio backend."""
        waveform, sr = torchaudio.load(filepath)

        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        return waveform.squeeze(0).numpy(), sr

    def _load_with_soundfile(
        self,
        filepath: Path,
        target_sr: int,
        mono: bool
    ) -> Tuple[np.ndarray, int]:
        """Load using soundfile backend."""
        audio, sr = sf.read(filepath)

        # Handle stereo
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Simple resampling
        if sr != target_sr:
            ratio = target_sr / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
            sr = target_sr

        return audio, sr

    def load_batch(
        self,
        filepaths: List[Union[str, Path]],
        sample_rate: Optional[int] = None,
        max_duration: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Load multiple audio files.

        Args:
            filepaths: List of audio file paths
            sample_rate: Target sample rate
            max_duration: Maximum duration in seconds (trim if longer)

        Returns:
            List of audio arrays
        """
        audios = []
        sr = sample_rate or self.config.sample_rate

        for filepath in filepaths:
            audio, _ = self.load(filepath, sr)

            if max_duration is not None:
                max_samples = int(max_duration * sr)
                audio = audio[:max_samples]

            audios.append(audio)

        return audios

    def save(
        self,
        filepath: Union[str, Path],
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ):
        """
        Save audio to file.

        Args:
            filepath: Output path
            audio: Audio array
            sample_rate: Sample rate
        """
        filepath = Path(filepath)
        sr = sample_rate or self.config.sample_rate

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if SOUNDFILE_AVAILABLE:
            sf.write(filepath, audio, sr)
        elif TORCH_AVAILABLE:
            tensor = torch.from_numpy(audio).unsqueeze(0)
            torchaudio.save(filepath, tensor, sr)
        else:
            raise ImportError("No audio backend available for saving.")

    def validate(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Validate audio array and return statistics.

        Args:
            audio: Audio array
            sample_rate: Sample rate

        Returns:
            Dictionary of audio statistics
        """
        duration = len(audio) / sample_rate

        stats = {
            'duration': duration,
            'sample_rate': sample_rate,
            'num_samples': len(audio),
            'min_value': float(np.min(audio)),
            'max_value': float(np.max(audio)),
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'has_clipping': bool(np.any(np.abs(audio) > 0.99)),
            'is_silent': bool(np.max(np.abs(audio)) < 0.01),
        }

        return stats


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for TTS training.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.loader = AudioLoader(config)

    def normalize(
        self,
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
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)

        return np.clip(audio * gain_linear, -1.0, 1.0).astype(np.float32)

    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: float = 20.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim leading and trailing silence.

        Args:
            audio: Input audio
            top_db: Threshold below peak in dB
            frame_length: Frame length for energy calculation
            hop_length: Hop between frames

        Returns:
            Trimmed audio
        """
        if LIBROSA_AVAILABLE:
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            return trimmed

        # Manual implementation
        energy = np.array([
            np.sum(audio[i:i + frame_length] ** 2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])

        threshold = np.max(energy) * (10 ** (-top_db / 10))
        non_silent = energy > threshold

        if not np.any(non_silent):
            return audio

        start_frame = np.argmax(non_silent)
        end_frame = len(non_silent) - np.argmax(non_silent[::-1])

        start_sample = start_frame * hop_length
        end_sample = min(end_frame * hop_length + frame_length, len(audio))

        return audio[start_sample:end_sample]

    def preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter.

        Args:
            audio: Input audio
            coef: Pre-emphasis coefficient

        Returns:
            Pre-emphasized audio
        """
        return np.append(audio[0], audio[1:] - coef * audio[:-1]).astype(np.float32)

    def deemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Remove pre-emphasis filter.

        Args:
            audio: Pre-emphasized audio
            coef: Pre-emphasis coefficient

        Returns:
            De-emphasized audio
        """
        result = np.zeros_like(audio)
        result[0] = audio[0]

        for i in range(1, len(audio)):
            result[i] = audio[i] + coef * result[i - 1]

        return result

    def process(
        self,
        audio: np.ndarray,
        normalize: bool = True,
        trim: bool = True,
        preemphasis: bool = False,
        target_db: float = -20.0
    ) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            audio: Input audio
            normalize: Apply normalization
            trim: Trim silence
            preemphasis: Apply pre-emphasis
            target_db: Target dB for normalization

        Returns:
            Processed audio
        """
        if trim:
            audio = self.trim_silence(audio)

        if normalize:
            audio = self.normalize(audio, target_db)

        if preemphasis:
            audio = self.preemphasis(audio)

        return audio

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Load, process, and optionally save audio file.

        Args:
            input_path: Input file path
            output_path: Output file path (optional)
            **kwargs: Arguments for process()

        Returns:
            Processed audio
        """
        audio, sr = self.loader.load(input_path)
        audio = self.process(audio, **kwargs)

        if output_path is not None:
            self.loader.save(output_path, audio, sr)

        return audio
