"""
Text Tokenizer Module
=====================
Tokenizes text for TTS model input.
"""

import re
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class TokenizerConfig:
    """Configuration for text tokenization."""
    # Character set type
    char_type: str = 'phoneme'  # 'phoneme', 'character', or 'bpe'

    # Special tokens
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'
    bos_token: str = '<bos>'
    eos_token: str = '<eos>'
    space_token: str = '<space>'

    # Processing options
    lowercase: bool = True
    add_bos: bool = True
    add_eos: bool = True

    # For character-based tokenization
    allowed_chars: str = "abcdefghijklmnopqrstuvwxyz'-. "


# Standard character sets
ENGLISH_CHARS = "abcdefghijklmnopqrstuvwxyz'-. !?,;:"
HINDI_CHARS = "अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक़ख़ग़ज़फ़ड़ढ़ँंःािीुूेैोौ्ृ। "

# IPA phoneme set (commonly used in TTS)
IPA_PHONEMES = [
    # Special tokens
    '<pad>', '<unk>', '<bos>', '<eos>', '<space>',

    # Consonants
    'p', 'b', 't', 'd', 'k', 'g', 'ʔ',
    'pʰ', 'bʰ', 'tʰ', 'dʰ', 'kʰ', 'gʰ',
    'm', 'n', 'ɲ', 'ŋ', 'ɳ',
    'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'h', 'ɦ', 'x', 'ɣ',
    'tʃ', 'dʒ', 'tʃʰ', 'dʒʰ',
    'ʈ', 'ɖ', 'ʈʰ', 'ɖʰ',
    't̪', 'd̪', 't̪ʰ', 'd̪ʰ',
    'l', 'ɫ', 'r', 'ɹ', 'ɾ', 'ɽ', 'ɽʰ',
    'j', 'w', 'ʋ',

    # Vowels
    'i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ɑ',
    'ə', 'ʌ', 'ɔ', 'o', 'ʊ', 'u',
    'iː', 'eː', 'ɛː', 'aː', 'ɑː', 'ɔː', 'oː', 'uː',

    # Diphthongs
    'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ',

    # Stress and length
    'ˈ', 'ˌ', 'ː',

    # Nasalization
    '̃',

    # Punctuation
    '.', ',', '!', '?', ';', ':', '-', "'", '"',
]

# ARPABET (alternative phoneme set)
ARPABET_PHONEMES = [
    '<pad>', '<unk>', '<bos>', '<eos>', '<space>',
    'AA', 'AA0', 'AA1', 'AA2',
    'AE', 'AE0', 'AE1', 'AE2',
    'AH', 'AH0', 'AH1', 'AH2',
    'AO', 'AO0', 'AO1', 'AO2',
    'AW', 'AW0', 'AW1', 'AW2',
    'AY', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH',
    'EH', 'EH0', 'EH1', 'EH2',
    'ER', 'ER0', 'ER1', 'ER2',
    'EY', 'EY0', 'EY1', 'EY2',
    'F', 'G', 'HH',
    'IH', 'IH0', 'IH1', 'IH2',
    'IY', 'IY0', 'IY1', 'IY2',
    'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW', 'OW0', 'OW1', 'OW2',
    'OY', 'OY0', 'OY1', 'OY2',
    'P', 'R', 'S', 'SH', 'T', 'TH',
    'UH', 'UH0', 'UH1', 'UH2',
    'UW', 'UW0', 'UW1', 'UW2',
    'V', 'W', 'Y', 'Z', 'ZH',
    '.', ',', '!', '?',
]


class TextTokenizer:
    """
    Tokenizer for converting text/phonemes to model input.

    Supports:
    - Character-based tokenization
    - Phoneme-based tokenization
    - Custom vocabularies
    """

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        vocab: Optional[List[str]] = None
    ):
        self.config = config or TokenizerConfig()

        # Build vocabulary
        if vocab is not None:
            self._build_vocab(vocab)
        else:
            self._build_default_vocab()

    def _build_vocab(self, vocab: List[str]):
        """Build vocabulary from list."""
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}

    def _build_default_vocab(self):
        """Build default vocabulary based on config."""
        if self.config.char_type == 'phoneme':
            self._build_vocab(IPA_PHONEMES)
        elif self.config.char_type == 'arpabet':
            self._build_vocab(ARPABET_PHONEMES)
        else:
            # Character-based
            special = [
                self.config.pad_token,
                self.config.unk_token,
                self.config.bos_token,
                self.config.eos_token,
                self.config.space_token,
            ]
            chars = list(self.config.allowed_chars)
            self._build_vocab(special + chars)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        """Return padding token ID."""
        return self.token_to_id.get(self.config.pad_token, 0)

    @property
    def unk_id(self) -> int:
        """Return unknown token ID."""
        return self.token_to_id.get(self.config.unk_token, 1)

    @property
    def bos_id(self) -> int:
        """Return beginning-of-sequence token ID."""
        return self.token_to_id.get(self.config.bos_token, 2)

    @property
    def eos_id(self) -> int:
        """Return end-of-sequence token ID."""
        return self.token_to_id.get(self.config.eos_token, 3)

    @property
    def space_id(self) -> int:
        """Return space token ID."""
        return self.token_to_id.get(self.config.space_token, 4)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if self.config.lowercase:
            text = text.lower()

        if self.config.char_type == 'phoneme':
            tokens = self._tokenize_phonemes(text)
        elif self.config.char_type == 'arpabet':
            tokens = self._tokenize_arpabet(text)
        else:
            tokens = self._tokenize_chars(text)

        return tokens

    def _tokenize_chars(self, text: str) -> List[str]:
        """Character-based tokenization."""
        tokens = []

        for char in text:
            if char == ' ':
                tokens.append(self.config.space_token)
            elif char in self.token_to_id:
                tokens.append(char)
            else:
                tokens.append(self.config.unk_token)

        return tokens

    def _tokenize_phonemes(self, text: str) -> List[str]:
        """IPA phoneme tokenization."""
        tokens = []
        i = 0

        while i < len(text):
            # Try multi-character phonemes first (longest match)
            matched = False

            for length in range(4, 0, -1):
                if i + length <= len(text):
                    substr = text[i:i + length]

                    if substr in self.token_to_id:
                        tokens.append(substr)
                        i += length
                        matched = True
                        break

            if not matched:
                char = text[i]
                if char == ' ':
                    tokens.append(self.config.space_token)
                elif char in self.token_to_id:
                    tokens.append(char)
                else:
                    tokens.append(self.config.unk_token)
                i += 1

        return tokens

    def _tokenize_arpabet(self, text: str) -> List[str]:
        """ARPABET tokenization (space-separated phonemes)."""
        tokens = []

        for token in text.split():
            if token in self.token_to_id:
                tokens.append(token)
            else:
                tokens.append(self.config.unk_token)

        return tokens

    def encode(
        self,
        text: str,
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Add BOS token (uses config default if None)
            add_eos: Add EOS token (uses config default if None)

        Returns:
            List of token IDs
        """
        if add_bos is None:
            add_bos = self.config.add_bos
        if add_eos is None:
            add_eos = self.config.add_eos

        tokens = self.tokenize(text)
        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special: Skip special tokens

        Returns:
            Decoded text
        """
        tokens = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id}

        for id_ in ids:
            if skip_special and id_ in special_ids:
                continue

            if id_ == self.space_id:
                tokens.append(' ')
            elif id_ in self.id_to_token:
                tokens.append(self.id_to_token[id_])
            else:
                tokens.append(self.config.unk_token)

        return ''.join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        return_lengths: bool = True
    ) -> Dict[str, Union[List[List[int]], List[int]]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts
            padding: Pad to same length
            max_length: Maximum sequence length
            return_lengths: Return sequence lengths

        Returns:
            Dictionary with 'input_ids' and optionally 'lengths'
        """
        encoded = [self.encode(text) for text in texts]
        lengths = [len(e) for e in encoded]

        if max_length is not None:
            encoded = [e[:max_length] for e in encoded]
            lengths = [min(l, max_length) for l in lengths]

        if padding:
            max_len = max(lengths)
            encoded = [
                e + [self.pad_id] * (max_len - len(e))
                for e in encoded
            ]

        result = {'input_ids': encoded}

        if return_lengths:
            result['lengths'] = lengths

        return result

    def save(self, path: Union[str, Path]):
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'vocab': self.vocab,
            'config': {
                'char_type': self.config.char_type,
                'pad_token': self.config.pad_token,
                'unk_token': self.config.unk_token,
                'bos_token': self.config.bos_token,
                'eos_token': self.config.eos_token,
                'space_token': self.config.space_token,
                'lowercase': self.config.lowercase,
                'add_bos': self.config.add_bos,
                'add_eos': self.config.add_eos,
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TextTokenizer':
        """Load tokenizer from file."""
        path = Path(path)

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = TokenizerConfig(**data['config'])
        return cls(config=config, vocab=data['vocab'])


class CharacterTokenizer(TextTokenizer):
    """Character-level tokenizer with language support."""

    def __init__(
        self,
        language: str = 'en',
        **kwargs
    ):
        # Set allowed characters based on language
        if language == 'hi':
            allowed_chars = HINDI_CHARS
        elif language == 'en':
            allowed_chars = ENGLISH_CHARS
        else:
            # Mixed/multilingual
            allowed_chars = ENGLISH_CHARS + HINDI_CHARS

        config = TokenizerConfig(
            char_type='character',
            allowed_chars=allowed_chars,
            **kwargs
        )

        super().__init__(config=config)


class PhonemeTokenizer(TextTokenizer):
    """IPA phoneme tokenizer."""

    def __init__(self, **kwargs):
        config = TokenizerConfig(char_type='phoneme', **kwargs)
        super().__init__(config=config)


# Factory functions
def create_tokenizer(
    tokenizer_type: str = 'character',
    language: str = 'en',
    **kwargs
) -> TextTokenizer:
    """
    Create a tokenizer.

    Args:
        tokenizer_type: 'character', 'phoneme', or 'arpabet'
        language: Language code ('en', 'hi')
        **kwargs: Additional config options

    Returns:
        Configured tokenizer
    """
    if tokenizer_type == 'character':
        return CharacterTokenizer(language=language, **kwargs)
    elif tokenizer_type == 'phoneme':
        return PhonemeTokenizer(**kwargs)
    elif tokenizer_type == 'arpabet':
        config = TokenizerConfig(char_type='arpabet', **kwargs)
        return TextTokenizer(config=config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
