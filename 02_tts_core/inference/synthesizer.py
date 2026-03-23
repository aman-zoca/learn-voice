"""
TTS Synthesizer
===============
Main synthesis pipeline that brings together all TTS components.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass

from ..preprocessing.text.normalizer import TextNormalizer, create_normalizer
from ..preprocessing.text.phonemizer import Phonemizer, create_phonemizer
from ..preprocessing.text.tokenizer import TextTokenizer, create_tokenizer
from ..preprocessing.audio.feature_extractor import MelSpectrogramExtractor, GriffinLimVocoder


@dataclass
class SynthesizerConfig:
    """Configuration for the synthesizer."""
    # Model paths
    acoustic_model_path: Optional[str] = None
    vocoder_path: Optional[str] = None

    # Model type
    model_type: str = 'tacotron2'  # 'tacotron2', 'vits', 'fastspeech'
    vocoder_type: str = 'hifigan'  # 'hifigan', 'griffin_lim'

    # Language
    language: str = 'en'

    # Audio settings
    sample_rate: int = 22050
    n_mels: int = 80

    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'


class Synthesizer:
    """
    Main TTS synthesizer class.

    Combines text preprocessing, acoustic model, and vocoder
    into a unified synthesis pipeline.
    """

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
        acoustic_model: Optional[torch.nn.Module] = None,
        vocoder: Optional[torch.nn.Module] = None
    ):
        self.config = config or SynthesizerConfig()

        # Set device
        self.device = self._get_device()

        # Initialize text processing
        self.normalizer = create_normalizer(self.config.language)
        self.phonemizer = create_phonemizer(
            language=self.config.language,
            backend='simple'  # Use simple backend by default
        )
        self.tokenizer = create_tokenizer(
            tokenizer_type='character',
            language=self.config.language
        )

        # Initialize models
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder

        # Load models if paths provided
        if self.config.acoustic_model_path:
            self.load_acoustic_model(self.config.acoustic_model_path)

        if self.config.vocoder_path:
            self.load_vocoder(self.config.vocoder_path)

        # Fallback to Griffin-Lim if no vocoder
        if self.vocoder is None and self.config.vocoder_type == 'griffin_lim':
            self.griffin_lim = GriffinLimVocoder()
        else:
            self.griffin_lim = None

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.config.device)

    def load_acoustic_model(self, path: Union[str, Path]):
        """
        Load acoustic model from checkpoint.

        Args:
            path: Path to model checkpoint
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Determine model type and load
        if self.config.model_type == 'tacotron2':
            from ..models.acoustic_model.tacotron2 import Tacotron2

            model_config = checkpoint.get('config', {})
            self.acoustic_model = Tacotron2(
                vocab_size=self.tokenizer.vocab_size,
                **model_config
            )

        elif self.config.model_type == 'vits':
            from ..models.vits.model import VITS

            model_config = checkpoint.get('config', {})
            self.acoustic_model = VITS(
                vocab_size=self.tokenizer.vocab_size,
                **model_config
            )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.acoustic_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.acoustic_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.acoustic_model.load_state_dict(checkpoint)

        self.acoustic_model.to(self.device)
        self.acoustic_model.eval()

    def load_vocoder(self, path: Union[str, Path]):
        """
        Load vocoder from checkpoint.

        Args:
            path: Path to vocoder checkpoint
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Vocoder not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        if self.config.vocoder_type == 'hifigan':
            from ..models.vocoder.hifigan import HiFiGANGenerator

            vocoder_config = checkpoint.get('config', {})
            self.vocoder = HiFiGANGenerator(**vocoder_config)

            if 'generator' in checkpoint:
                self.vocoder.load_state_dict(checkpoint['generator'])
            elif 'model_state_dict' in checkpoint:
                self.vocoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.vocoder.load_state_dict(checkpoint)

            self.vocoder.to(self.device)
            self.vocoder.eval()

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Preprocess text for synthesis.

        Args:
            text: Input text

        Returns:
            Dictionary with processed text data
        """
        # Normalize text
        normalized = self.normalizer.normalize(text)

        # Convert to phonemes (optional)
        phonemes = self.phonemizer.phonemize(normalized)

        # Tokenize
        tokens = self.tokenizer.encode(phonemes)

        return {
            'original': text,
            'normalized': normalized,
            'phonemes': phonemes,
            'tokens': tokens
        }

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Input text
            speaker_id: Speaker ID for multi-speaker models
            speed: Speed factor (1.0 = normal)
            pitch_shift: Pitch shift in semitones

        Returns:
            Audio waveform as numpy array
        """
        # Preprocess text
        processed = self.preprocess_text(text)
        tokens = processed['tokens']

        # Convert to tensor
        text_tensor = torch.LongTensor([tokens]).to(self.device)
        text_lengths = torch.LongTensor([len(tokens)]).to(self.device)

        # Generate mel spectrogram or audio
        if self.config.model_type == 'vits':
            # VITS is end-to-end
            if speaker_id is not None:
                sid = torch.LongTensor([speaker_id]).to(self.device)
            else:
                sid = None

            audio = self.acoustic_model.infer(
                text_tensor,
                text_lengths,
                sid=sid,
                length_scale=1.0 / speed
            )

            audio = audio.squeeze(0).squeeze(0).cpu().numpy()

        else:
            # Two-stage: acoustic model + vocoder
            outputs = self.acoustic_model.inference(text_tensor, text_lengths)
            mel = outputs['mel_outputs_postnet']

            # Vocoder
            if self.vocoder is not None:
                audio = self.vocoder(mel)
                audio = audio.squeeze(0).squeeze(0).cpu().numpy()
            else:
                # Fallback to Griffin-Lim
                mel_np = mel.squeeze(0).cpu().numpy()
                # Denormalize mel if needed
                # mel_np = denormalize_mel(mel_np)
                audio = self.griffin_lim(np.exp(mel_np))

        # Apply pitch shift if needed
        if pitch_shift != 0.0:
            audio = self._pitch_shift(audio, pitch_shift)

        return audio

    def synthesize_batch(
        self,
        texts: List[str],
        speaker_ids: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Synthesize multiple texts.

        Args:
            texts: List of input texts
            speaker_ids: List of speaker IDs

        Returns:
            List of audio waveforms
        """
        audios = []

        for i, text in enumerate(texts):
            sid = speaker_ids[i] if speaker_ids else None
            audio = self.synthesize(text, speaker_id=sid)
            audios.append(audio)

        return audios

    def _pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """
        Shift pitch of audio.

        Args:
            audio: Input audio
            semitones: Shift amount in semitones

        Returns:
            Pitch-shifted audio
        """
        try:
            import librosa
            return librosa.effects.pitch_shift(
                audio, sr=self.config.sample_rate, n_steps=semitones
            )
        except ImportError:
            # Return unchanged if librosa not available
            return audio

    def save_audio(
        self,
        audio: np.ndarray,
        path: Union[str, Path],
        sample_rate: Optional[int] = None
    ):
        """
        Save audio to file.

        Args:
            audio: Audio waveform
            path: Output path
            sample_rate: Sample rate (uses config if None)
        """
        path = Path(path)
        sr = sample_rate or self.config.sample_rate

        try:
            import soundfile as sf
            sf.write(path, audio, sr)
        except ImportError:
            import scipy.io.wavfile as wav
            # Convert to int16 for scipy
            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(path, sr, audio_int16)


class StreamingSynthesizer(Synthesizer):
    """
    Streaming synthesizer for real-time synthesis.

    Generates audio in chunks for lower latency.
    """

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
        chunk_size: int = 8192
    ):
        super().__init__(config)
        self.chunk_size = chunk_size

    def synthesize_stream(
        self,
        text: str,
        speaker_id: Optional[int] = None
    ):
        """
        Generate audio in chunks.

        Yields:
            Audio chunks as numpy arrays
        """
        # For now, just yield the full audio in chunks
        # A proper implementation would use incremental decoding
        audio = self.synthesize(text, speaker_id)

        for i in range(0, len(audio), self.chunk_size):
            yield audio[i:i + self.chunk_size]


# Factory function
def create_synthesizer(
    model_type: str = 'tacotron2',
    language: str = 'en',
    acoustic_model_path: Optional[str] = None,
    vocoder_path: Optional[str] = None,
    device: str = 'auto'
) -> Synthesizer:
    """
    Create a synthesizer.

    Args:
        model_type: Type of TTS model
        language: Language code
        acoustic_model_path: Path to acoustic model
        vocoder_path: Path to vocoder
        device: Compute device

    Returns:
        Configured synthesizer
    """
    config = SynthesizerConfig(
        model_type=model_type,
        language=language,
        acoustic_model_path=acoustic_model_path,
        vocoder_path=vocoder_path,
        device=device
    )

    return Synthesizer(config)


def main():
    """CLI entry point for synthesis."""
    import argparse

    parser = argparse.ArgumentParser(description='TTS Synthesizer')
    parser.add_argument('text', type=str, help='Text to synthesize')
    parser.add_argument('--output', '-o', type=str, default='output.wav',
                        help='Output audio file')
    parser.add_argument('--model', type=str, default='tacotron2',
                        help='Model type')
    parser.add_argument('--language', type=str, default='en',
                        help='Language')
    parser.add_argument('--acoustic-model', type=str,
                        help='Path to acoustic model')
    parser.add_argument('--vocoder', type=str,
                        help='Path to vocoder')
    parser.add_argument('--speaker', type=int, default=None,
                        help='Speaker ID')

    args = parser.parse_args()

    # Create synthesizer
    synth = create_synthesizer(
        model_type=args.model,
        language=args.language,
        acoustic_model_path=args.acoustic_model,
        vocoder_path=args.vocoder
    )

    # Synthesize
    audio = synth.synthesize(args.text, speaker_id=args.speaker)

    # Save
    synth.save_audio(audio, args.output)
    print(f"Saved audio to {args.output}")


if __name__ == '__main__':
    main()
