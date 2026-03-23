"""
Phonemizer Module
=================
Converts text to phonemes for TTS input.
"""

import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

try:
    import epitran
    EPITRAN_AVAILABLE = True
except ImportError:
    EPITRAN_AVAILABLE = False

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False


@dataclass
class PhonemizeConfig:
    """Configuration for phonemization."""
    language: str = 'en-us'
    backend: str = 'espeak'  # 'espeak', 'epitran', or 'simple'
    preserve_punctuation: bool = True
    strip: bool = True
    with_stress: bool = True


# Simple English grapheme-to-phoneme rules (fallback)
ENGLISH_G2P: Dict[str, str] = {
    'a': 'æ', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɛ',
    'f': 'f', 'g': 'g', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ',
    'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'ɑ',
    'p': 'p', 'q': 'k', 'r': 'r', 's': 's', 't': 't',
    'u': 'ʌ', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j',
    'z': 'z'
}

# Common English word pronunciations
ENGLISH_LEXICON: Dict[str, str] = {
    'the': 'ðə',
    'a': 'eɪ',
    'an': 'æn',
    'is': 'ɪz',
    'are': 'ɑr',
    'was': 'wʌz',
    'were': 'wɜr',
    'be': 'bi',
    'been': 'bɪn',
    'have': 'hæv',
    'has': 'hæz',
    'had': 'hæd',
    'do': 'du',
    'does': 'dʌz',
    'did': 'dɪd',
    'will': 'wɪl',
    'would': 'wʊd',
    'could': 'kʊd',
    'should': 'ʃʊd',
    'may': 'meɪ',
    'might': 'maɪt',
    'must': 'mʌst',
    'can': 'kæn',
    'to': 'tu',
    'of': 'ʌv',
    'in': 'ɪn',
    'for': 'fɔr',
    'on': 'ɑn',
    'with': 'wɪð',
    'at': 'æt',
    'by': 'baɪ',
    'from': 'frʌm',
    'up': 'ʌp',
    'about': 'əˈbaʊt',
    'into': 'ˈɪntu',
    'over': 'ˈoʊvər',
    'after': 'ˈæftər',
    'i': 'aɪ',
    'you': 'ju',
    'he': 'hi',
    'she': 'ʃi',
    'it': 'ɪt',
    'we': 'wi',
    'they': 'ðeɪ',
    'this': 'ðɪs',
    'that': 'ðæt',
    'these': 'ðiz',
    'those': 'ðoʊz',
    'hello': 'həˈloʊ',
    'world': 'wɜrld',
}

# Hindi phoneme inventory
HINDI_VOWELS = {
    'अ': 'ə', 'आ': 'aː', 'इ': 'ɪ', 'ई': 'iː',
    'उ': 'ʊ', 'ऊ': 'uː', 'ए': 'eː', 'ऐ': 'æː',
    'ओ': 'oː', 'औ': 'ɔː', 'ऋ': 'r̩',
}

HINDI_MATRAS = {
    'ा': 'aː', 'ि': 'ɪ', 'ी': 'iː', 'ु': 'ʊ',
    'ू': 'uː', 'े': 'eː', 'ै': 'æː', 'ो': 'oː',
    'ौ': 'ɔː', 'ृ': 'r̩', '्': '',  # Halant
}

HINDI_CONSONANTS = {
    'क': 'k', 'ख': 'kʰ', 'ग': 'g', 'घ': 'gʰ', 'ङ': 'ŋ',
    'च': 'tʃ', 'छ': 'tʃʰ', 'ज': 'dʒ', 'झ': 'dʒʰ', 'ञ': 'ɲ',
    'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʰ', 'ण': 'ɳ',
    'त': 't', 'थ': 'tʰ', 'द': 'd', 'ध': 'dʰ', 'न': 'n',
    'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʰ', 'म': 'm',
    'य': 'j', 'र': 'r', 'ल': 'l', 'व': 'ʋ',
    'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'ɦ',
    'क़': 'q', 'ख़': 'x', 'ग़': 'ɣ', 'ज़': 'z', 'फ़': 'f',
    'ड़': 'ɽ', 'ढ़': 'ɽʰ',
}


class Phonemizer:
    """
    Convert text to phonemes.

    Supports multiple backends:
    - espeak: Most accurate, requires espeak-ng
    - epitran: IPA transcription, good for multilingual
    - simple: Basic rule-based, no dependencies
    """

    def __init__(self, config: Optional[PhonemizeConfig] = None):
        self.config = config or PhonemizeConfig()
        self._init_backend()

    def _init_backend(self):
        """Initialize the phonemization backend."""
        if self.config.backend == 'espeak' and PHONEMIZER_AVAILABLE:
            self.backend = 'espeak'
        elif self.config.backend == 'epitran' and EPITRAN_AVAILABLE:
            self.backend = 'epitran'
            # Map language codes
            epi_lang = {
                'en-us': 'eng-Latn',
                'en-gb': 'eng-Latn',
                'en': 'eng-Latn',
                'hi': 'hin-Deva',
            }.get(self.config.language, 'eng-Latn')
            try:
                self.epi = epitran.Epitran(epi_lang)
            except Exception:
                self.backend = 'simple'
        else:
            self.backend = 'simple'

    def phonemize(self, text: str) -> str:
        """
        Convert text to phonemes.

        Args:
            text: Input text

        Returns:
            Phoneme string
        """
        if self.backend == 'espeak':
            return self._phonemize_espeak(text)
        elif self.backend == 'epitran':
            return self._phonemize_epitran(text)
        else:
            return self._phonemize_simple(text)

    def _phonemize_espeak(self, text: str) -> str:
        """Phonemize using espeak-ng."""
        result = phonemize(
            text,
            language=self.config.language,
            backend='espeak',
            strip=self.config.strip,
            preserve_punctuation=self.config.preserve_punctuation,
            with_stress=self.config.with_stress
        )
        return result

    def _phonemize_epitran(self, text: str) -> str:
        """Phonemize using epitran."""
        # Handle punctuation
        if self.config.preserve_punctuation:
            # Split into words and punctuation
            tokens = re.findall(r'\w+|[^\w\s]', text)
            result = []

            for token in tokens:
                if re.match(r'\w+', token):
                    result.append(self.epi.transliterate(token))
                else:
                    result.append(token)

            return ' '.join(result)
        else:
            return self.epi.transliterate(text)

    def _phonemize_simple(self, text: str) -> str:
        """Simple rule-based phonemization."""
        # Determine language
        if self._is_hindi(text):
            return self._phonemize_hindi(text)
        else:
            return self._phonemize_english(text)

    def _is_hindi(self, text: str) -> bool:
        """Check if text is primarily Hindi."""
        hindi_chars = re.findall(r'[\u0900-\u097F]', text)
        return len(hindi_chars) > len(text) // 2

    def _phonemize_english(self, text: str) -> str:
        """Simple English phonemization."""
        text = text.lower()

        # Handle punctuation
        punct_map = {
            '.': '.',
            ',': ',',
            '!': '!',
            '?': '?',
            ';': ';',
            ':': ':',
            '-': ' ',
            "'": '',
            '"': '',
        }

        # Split into tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        result = []

        for token in tokens:
            if token in punct_map:
                if self.config.preserve_punctuation:
                    result.append(punct_map[token])
            elif token in ENGLISH_LEXICON:
                result.append(ENGLISH_LEXICON[token])
            else:
                # Character-by-character fallback
                phonemes = []
                for char in token:
                    if char in ENGLISH_G2P:
                        phonemes.append(ENGLISH_G2P[char])
                result.append(''.join(phonemes))

        return ' '.join(result)

    def _phonemize_hindi(self, text: str) -> str:
        """Simple Hindi phonemization using Devanagari rules."""
        result = []
        i = 0

        while i < len(text):
            char = text[i]

            # Check for consonant
            if char in HINDI_CONSONANTS:
                phoneme = HINDI_CONSONANTS[char]

                # Check for matra (vowel sign)
                if i + 1 < len(text) and text[i + 1] in HINDI_MATRAS:
                    matra = text[i + 1]
                    if matra == '्':  # Halant - no inherent vowel
                        result.append(phoneme)
                    else:
                        result.append(phoneme + HINDI_MATRAS[matra])
                    i += 2
                else:
                    # Add inherent 'a' vowel
                    result.append(phoneme + 'ə')
                    i += 1

            # Check for independent vowel
            elif char in HINDI_VOWELS:
                result.append(HINDI_VOWELS[char])
                i += 1

            # Check for punctuation
            elif char in '।,;:!?.':
                if self.config.preserve_punctuation:
                    result.append(char if char != '।' else '.')
                i += 1

            # Whitespace
            elif char.isspace():
                result.append(' ')
                i += 1

            else:
                i += 1

        return ''.join(result)

    def phonemize_batch(self, texts: List[str]) -> List[str]:
        """
        Phonemize multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of phoneme strings
        """
        return [self.phonemize(text) for text in texts]


class PhonemeTokenizer:
    """
    Tokenize phoneme strings into integer indices.

    This is needed to convert phonemes to model input.
    """

    def __init__(self, phoneme_list: Optional[List[str]] = None):
        if phoneme_list is None:
            phoneme_list = self._default_phoneme_list()

        self.phoneme_to_id = {p: i for i, p in enumerate(phoneme_list)}
        self.id_to_phoneme = {i: p for i, p in enumerate(phoneme_list)}

        # Special tokens
        self.pad_id = self.phoneme_to_id.get('<pad>', 0)
        self.unk_id = self.phoneme_to_id.get('<unk>', 1)
        self.bos_id = self.phoneme_to_id.get('<bos>', 2)
        self.eos_id = self.phoneme_to_id.get('<eos>', 3)

    def _default_phoneme_list(self) -> List[str]:
        """Default phoneme vocabulary."""
        special = ['<pad>', '<unk>', '<bos>', '<eos>', ' ']

        # IPA consonants
        consonants = [
            'p', 'b', 't', 'd', 'k', 'g', 'q',
            'pʰ', 'tʰ', 'kʰ', 'bʰ', 'dʰ', 'gʰ',
            'm', 'n', 'ŋ', 'ɲ', 'ɳ',
            'f', 'v', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'h', 'ɦ', 'x', 'ɣ',
            'θ', 'ð',
            'tʃ', 'dʒ', 'tʃʰ', 'dʒʰ',
            'l', 'r', 'ɹ', 'j', 'w', 'ʋ',
            'ʈ', 'ɖ', 'ʈʰ', 'ɖʰ', 'ɽ', 'ɽʰ',
        ]

        # IPA vowels
        vowels = [
            'i', 'ɪ', 'e', 'ɛ', 'æ', 'a',
            'ə', 'ʌ', 'ɔ', 'o', 'ʊ', 'u',
            'iː', 'eː', 'æː', 'aː', 'oː', 'ɔː', 'uː',
            'r̩',
            # Diphthongs
            'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ',
        ]

        # Stress markers
        stress = ['ˈ', 'ˌ']

        # Punctuation
        punctuation = ['.', ',', '!', '?', ';', ':', '-', "'"]

        return special + consonants + vowels + stress + punctuation

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.phoneme_to_id)

    def encode(self, phonemes: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert phoneme string to token IDs.

        Args:
            phonemes: Phoneme string
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token

        Returns:
            List of token IDs
        """
        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[phoneme])
            else:
                ids.append(self.unk_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to phoneme string.

        Args:
            ids: List of token IDs

        Returns:
            Phoneme string
        """
        phonemes = []

        for id_ in ids:
            if id_ in self.id_to_phoneme:
                phoneme = self.id_to_phoneme[id_]
                if phoneme not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    phonemes.append(phoneme)

        return ''.join(phonemes)

    def batch_encode(
        self,
        phoneme_list: List[str],
        add_bos: bool = False,
        add_eos: bool = False
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Encode a batch of phoneme strings.

        Args:
            phoneme_list: List of phoneme strings
            add_bos: Add BOS token
            add_eos: Add EOS token

        Returns:
            Tuple of (encoded_ids, lengths)
        """
        encoded = [self.encode(p, add_bos, add_eos) for p in phoneme_list]
        lengths = [len(e) for e in encoded]

        return encoded, lengths


# Factory functions
def create_phonemizer(language: str = 'en-us', backend: str = 'espeak') -> Phonemizer:
    """Create a phonemizer for the specified language."""
    config = PhonemizeConfig(language=language, backend=backend)
    return Phonemizer(config)


def create_tokenizer(phoneme_list: Optional[List[str]] = None) -> PhonemeTokenizer:
    """Create a phoneme tokenizer."""
    return PhonemeTokenizer(phoneme_list)
