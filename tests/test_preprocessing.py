"""
Tests for preprocessing modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTextNormalization:
    """Tests for text normalization."""

    def test_number_expansion(self):
        """Test number expansion."""
        from tts_core.preprocessing.text.normalizer import TextNormalizer, NormalizerConfig

        config = NormalizerConfig(expand_numbers=True)
        normalizer = TextNormalizer(config)

        # Basic numbers
        assert 'one' in normalizer.normalize("1").lower()
        assert 'two' in normalizer.normalize("2").lower()
        assert 'ten' in normalizer.normalize("10").lower()
        assert 'hundred' in normalizer.normalize("100").lower()

    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        from tts_core.preprocessing.text.normalizer import TextNormalizer, NormalizerConfig

        config = NormalizerConfig(expand_abbreviations=True)
        normalizer = TextNormalizer(config)

        text = "Dr. Smith went to St. Mary's."
        normalized = normalizer.normalize(text)

        assert 'dr.' not in normalized.lower()
        assert 'doctor' in normalized.lower()

    def test_whitespace_normalization(self):
        """Test whitespace handling."""
        from tts_core.preprocessing.text.normalizer import TextNormalizer, NormalizerConfig

        config = NormalizerConfig(remove_extra_spaces=True)
        normalizer = TextNormalizer(config)

        text = "Hello    world   !"
        normalized = normalizer.normalize(text)

        # Should not have double spaces
        assert '  ' not in normalized


class TestTokenizer:
    """Tests for text tokenization."""

    def test_character_tokenizer(self):
        """Test character tokenization."""
        from tts_core.preprocessing.text.tokenizer import CharacterTokenizer

        tokenizer = CharacterTokenizer(language='en')

        text = "hello"
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0
        assert isinstance(tokens, list)

    def test_encode_decode(self):
        """Test that decode inverts encode."""
        from tts_core.preprocessing.text.tokenizer import CharacterTokenizer

        tokenizer = CharacterTokenizer(language='en', add_bos=False, add_eos=False)

        text = "hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded.strip() == text

    def test_batch_encode(self):
        """Test batch encoding."""
        from tts_core.preprocessing.text.tokenizer import CharacterTokenizer

        tokenizer = CharacterTokenizer(language='en')

        texts = ["hello", "world", "test"]
        result = tokenizer.batch_encode(texts, padding=True)

        assert 'input_ids' in result
        assert 'lengths' in result
        assert len(result['input_ids']) == 3


class TestAudioProcessing:
    """Tests for audio processing."""

    def test_mel_spectrogram_shape(self):
        """Test mel spectrogram output shape."""
        from tts_core.preprocessing.audio.feature_extractor import MelSpectrogramExtractor

        extractor = MelSpectrogramExtractor()

        # Create dummy audio (1 second at 22050 Hz)
        audio = np.random.randn(22050).astype(np.float32)

        mel = extractor.mel_spectrogram(audio)

        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0  # time frames

    def test_mel_filterbank(self):
        """Test mel filterbank creation."""
        from tts_core.preprocessing.audio.feature_extractor import MelSpectrogramExtractor

        extractor = MelSpectrogramExtractor()
        mel_basis = extractor.mel_basis

        assert mel_basis.shape[0] == 80  # n_mels
        assert mel_basis.shape[1] == 1025  # n_fft // 2 + 1

    def test_griffin_lim(self):
        """Test Griffin-Lim reconstruction."""
        from tts_core.preprocessing.audio.feature_extractor import GriffinLimVocoder

        vocoder = GriffinLimVocoder(n_iter=10)

        # Create dummy magnitude spectrogram
        magnitude = np.random.rand(1025, 100).astype(np.float32) + 0.1

        audio = vocoder(magnitude)

        assert len(audio) > 0
        assert isinstance(audio, np.ndarray)


class TestPhonemes:
    """Tests for phonemization."""

    def test_simple_phonemizer(self):
        """Test simple phonemizer."""
        from tts_core.preprocessing.text.phonemizer import Phonemizer, PhonemizeConfig

        config = PhonemizeConfig(backend='simple')
        phonemizer = Phonemizer(config)

        text = "hello"
        phonemes = phonemizer.phonemize(text)

        assert len(phonemes) > 0
        assert isinstance(phonemes, str)

    def test_hindi_detection(self):
        """Test Hindi text detection."""
        from tts_core.preprocessing.text.phonemizer import Phonemizer

        phonemizer = Phonemizer()

        # English text
        assert not phonemizer._is_hindi("hello world")

        # Hindi text
        assert phonemizer._is_hindi("नमस्ते दुनिया")


# Fixtures
@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_mel():
    """Generate sample mel spectrogram."""
    return np.random.randn(80, 100).astype(np.float32)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
