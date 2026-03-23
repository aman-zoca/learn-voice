"""
Hindi Text Processor Module
===========================
Specialized text processing for Hindi language TTS.
"""

import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANS_AVAILABLE = True
except ImportError:
    INDIC_TRANS_AVAILABLE = False


@dataclass
class HindiProcessorConfig:
    """Configuration for Hindi text processing."""
    normalize_nukta: bool = True
    normalize_chandrabindu: bool = True
    handle_schwa_deletion: bool = True
    transliterate_english: bool = True
    expand_numbers: bool = True


# Hindi number words
HINDI_NUMBERS = {
    0: 'शून्य',
    1: 'एक',
    2: 'दो',
    3: 'तीन',
    4: 'चार',
    5: 'पाँच',
    6: 'छह',
    7: 'सात',
    8: 'आठ',
    9: 'नौ',
    10: 'दस',
    11: 'ग्यारह',
    12: 'बारह',
    13: 'तेरह',
    14: 'चौदह',
    15: 'पंद्रह',
    16: 'सोलह',
    17: 'सत्रह',
    18: 'अठारह',
    19: 'उन्नीस',
    20: 'बीस',
    21: 'इक्कीस',
    22: 'बाईस',
    23: 'तेईस',
    24: 'चौबीस',
    25: 'पच्चीस',
    26: 'छब्बीस',
    27: 'सत्ताईस',
    28: 'अट्ठाईस',
    29: 'उनतीस',
    30: 'तीस',
    31: 'इकतीस',
    32: 'बत्तीस',
    33: 'तैंतीस',
    34: 'चौंतीस',
    35: 'पैंतीस',
    36: 'छत्तीस',
    37: 'सैंतीस',
    38: 'अड़तीस',
    39: 'उनतालीस',
    40: 'चालीस',
    41: 'इकतालीस',
    42: 'बयालीस',
    43: 'तैंतालीस',
    44: 'चवालीस',
    45: 'पैंतालीस',
    46: 'छियालीस',
    47: 'सैंतालीस',
    48: 'अड़तालीस',
    49: 'उनचास',
    50: 'पचास',
    51: 'इक्यावन',
    52: 'बावन',
    53: 'तिरपन',
    54: 'चौवन',
    55: 'पचपन',
    56: 'छप्पन',
    57: 'सत्तावन',
    58: 'अट्ठावन',
    59: 'उनसठ',
    60: 'साठ',
    61: 'इकसठ',
    62: 'बासठ',
    63: 'तिरसठ',
    64: 'चौंसठ',
    65: 'पैंसठ',
    66: 'छियासठ',
    67: 'सड़सठ',
    68: 'अड़सठ',
    69: 'उनहत्तर',
    70: 'सत्तर',
    71: 'इकहत्तर',
    72: 'बहत्तर',
    73: 'तिहत्तर',
    74: 'चौहत्तर',
    75: 'पचहत्तर',
    76: 'छिहत्तर',
    77: 'सतहत्तर',
    78: 'अठहत्तर',
    79: 'उनासी',
    80: 'अस्सी',
    81: 'इक्यासी',
    82: 'बयासी',
    83: 'तिरासी',
    84: 'चौरासी',
    85: 'पचासी',
    86: 'छियासी',
    87: 'सत्तासी',
    88: 'अट्ठासी',
    89: 'नवासी',
    90: 'नब्बे',
    91: 'इक्यानबे',
    92: 'बानबे',
    93: 'तिरानबे',
    94: 'चौरानबे',
    95: 'पचानबे',
    96: 'छियानबे',
    97: 'सत्तानबे',
    98: 'अट्ठानबे',
    99: 'निन्यानबे',
    100: 'सौ',
}

# Hindi abbreviations
HINDI_ABBREVIATIONS = {
    'डॉ': 'डॉक्टर',
    'श्री': 'श्रीमान',
    'श्रीमती': 'श्रीमती',
    'कु': 'कुमारी',
    'प्रो': 'प्रोफेसर',
    'सुश्री': 'सुश्री',
    'मि': 'मिस्टर',
    'मिस': 'मिस',
}

# Common English words with Hindi pronunciations
ENGLISH_TO_HINDI = {
    'computer': 'कंप्यूटर',
    'mobile': 'मोबाइल',
    'phone': 'फोन',
    'internet': 'इंटरनेट',
    'email': 'ईमेल',
    'website': 'वेबसाइट',
    'office': 'ऑफिस',
    'school': 'स्कूल',
    'college': 'कॉलेज',
    'university': 'यूनिवर्सिटी',
    'hospital': 'हॉस्पिटल',
    'doctor': 'डॉक्टर',
    'engineer': 'इंजीनियर',
    'bank': 'बैंक',
    'station': 'स्टेशन',
    'bus': 'बस',
    'train': 'ट्रेन',
    'car': 'कार',
    'taxi': 'टैक्सी',
    'hotel': 'होटल',
    'restaurant': 'रेस्टोरेंट',
}


class HindiTextProcessor:
    """
    Process Hindi text for TTS.

    Handles:
    - Unicode normalization
    - Number expansion
    - Abbreviation expansion
    - Schwa deletion rules
    - Mixed Hindi-English text
    """

    def __init__(self, config: Optional[HindiProcessorConfig] = None):
        self.config = config or HindiProcessorConfig()

        # Compile patterns
        self.number_pattern = re.compile(r'\d+')
        self.english_word_pattern = re.compile(r'[a-zA-Z]+')

    def process(self, text: str) -> str:
        """
        Full Hindi text processing pipeline.

        Args:
            text: Input text (can be mixed Hindi-English)

        Returns:
            Processed text
        """
        # Normalize Unicode
        text = self._normalize_unicode(text)

        # Expand abbreviations
        text = self._expand_abbreviations(text)

        # Handle English words
        if self.config.transliterate_english:
            text = self._handle_english(text)

        # Expand numbers
        if self.config.expand_numbers:
            text = self._expand_numbers(text)

        # Apply schwa deletion
        if self.config.handle_schwa_deletion:
            text = self._apply_schwa_deletion(text)

        # Clean up
        text = self._clean_text(text)

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Hindi Unicode characters."""
        # Normalize nukta characters
        if self.config.normalize_nukta:
            # Combine base + nukta to single character where possible
            nukta_map = {
                'क़': 'क़', 'ख़': 'ख़', 'ग़': 'ग़',
                'ज़': 'ज़', 'ड़': 'ड़', 'ढ़': 'ढ़',
                'फ़': 'फ़', 'य़': 'य़',
            }
            for k, v in nukta_map.items():
                text = text.replace(k, v)

        # Normalize chandrabindu
        if self.config.normalize_chandrabindu:
            # ँ (chandrabindu) can sometimes be replaced with ं (anusvara)
            # But keep it for better pronunciation
            pass

        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand Hindi abbreviations."""
        for abbrev, expansion in HINDI_ABBREVIATIONS.items():
            # Match abbreviation with optional period
            pattern = re.compile(re.escape(abbrev) + r'\.?')
            text = pattern.sub(expansion, text)

        return text

    def _handle_english(self, text: str) -> str:
        """Handle English words in Hindi text."""
        def replace_english(match):
            word = match.group(0).lower()

            # Check if we have a Hindi transliteration
            if word in ENGLISH_TO_HINDI:
                return ENGLISH_TO_HINDI[word]

            # Otherwise, attempt basic transliteration
            return self._transliterate_english_to_hindi(word)

        return self.english_word_pattern.sub(replace_english, text)

    def _transliterate_english_to_hindi(self, word: str) -> str:
        """Basic English to Hindi transliteration."""
        # Use indic_transliteration if available
        if INDIC_TRANS_AVAILABLE:
            try:
                return transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
            except Exception:
                pass

        # Basic mapping
        mapping = {
            'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ee': 'ई',
            'u': 'उ', 'oo': 'ऊ', 'e': 'ए', 'ai': 'ऐ',
            'o': 'ओ', 'au': 'औ',
            'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ',
            'ch': 'च', 'chh': 'छ', 'j': 'ज', 'jh': 'झ',
            't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध',
            'n': 'न', 'p': 'प', 'ph': 'फ', 'f': 'फ',
            'b': 'ब', 'bh': 'भ', 'm': 'म',
            'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
            'sh': 'श', 's': 'स', 'h': 'ह',
        }

        result = []
        i = 0
        while i < len(word):
            # Try two-character combinations first
            if i + 1 < len(word):
                two_char = word[i:i+2].lower()
                if two_char in mapping:
                    result.append(mapping[two_char])
                    i += 2
                    continue

            # Single character
            char = word[i].lower()
            if char in mapping:
                result.append(mapping[char])
            i += 1

        return ''.join(result)

    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to Hindi words."""
        def number_to_hindi(match):
            num_str = match.group(0)
            try:
                num = int(num_str)
                return self._int_to_hindi(num)
            except ValueError:
                return num_str

        return self.number_pattern.sub(number_to_hindi, text)

    def _int_to_hindi(self, num: int) -> str:
        """Convert integer to Hindi words."""
        if num < 0:
            return 'ऋण ' + self._int_to_hindi(-num)

        if num in HINDI_NUMBERS:
            return HINDI_NUMBERS[num]

        if num < 100:
            # Shouldn't happen if HINDI_NUMBERS is complete
            return str(num)

        if num < 1000:
            hundreds = num // 100
            remainder = num % 100

            result = HINDI_NUMBERS.get(hundreds, str(hundreds)) + ' सौ'
            if remainder > 0:
                result += ' ' + self._int_to_hindi(remainder)
            return result

        if num < 100000:  # Less than 1 lakh
            thousands = num // 1000
            remainder = num % 1000

            result = self._int_to_hindi(thousands) + ' हज़ार'
            if remainder > 0:
                result += ' ' + self._int_to_hindi(remainder)
            return result

        if num < 10000000:  # Less than 1 crore
            lakhs = num // 100000
            remainder = num % 100000

            result = self._int_to_hindi(lakhs) + ' लाख'
            if remainder > 0:
                result += ' ' + self._int_to_hindi(remainder)
            return result

        # Crores
        crores = num // 10000000
        remainder = num % 10000000

        result = self._int_to_hindi(crores) + ' करोड़'
        if remainder > 0:
            result += ' ' + self._int_to_hindi(remainder)
        return result

    def _apply_schwa_deletion(self, text: str) -> str:
        """
        Apply schwa deletion rules for Hindi.

        In Hindi, the inherent 'a' vowel (schwa) is often not pronounced
        at the end of words and in certain positions.

        This is a simplified implementation - full schwa deletion
        is context-dependent and complex.
        """
        # This is a placeholder - proper schwa deletion requires
        # morphological analysis
        return text

    def _clean_text(self, text: str) -> str:
        """Clean up text after processing."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove spaces before punctuation
        text = re.sub(r'\s+([।,;:!?])', r'\1', text)

        return text.strip()


class HindiPhonemizer:
    """
    Convert Hindi text to phonemes.

    Hindi has a mostly phonetic writing system, so conversion
    is relatively straightforward.
    """

    def __init__(self):
        # Devanagari to IPA mapping
        self.consonants = {
            'क': 'k', 'ख': 'kʰ', 'ग': 'g', 'घ': 'gʰ', 'ङ': 'ŋ',
            'च': 'tʃ', 'छ': 'tʃʰ', 'ज': 'dʒ', 'झ': 'dʒʰ', 'ञ': 'ɲ',
            'ट': 'ʈ', 'ठ': 'ʈʰ', 'ड': 'ɖ', 'ढ': 'ɖʰ', 'ण': 'ɳ',
            'त': 't̪', 'थ': 't̪ʰ', 'द': 'd̪', 'ध': 'd̪ʰ', 'न': 'n',
            'प': 'p', 'फ': 'pʰ', 'ब': 'b', 'भ': 'bʰ', 'म': 'm',
            'य': 'j', 'र': 'r', 'ल': 'l', 'व': 'ʋ',
            'श': 'ʃ', 'ष': 'ʂ', 'स': 's', 'ह': 'ɦ',
            # Nukta consonants
            'क़': 'q', 'ख़': 'x', 'ग़': 'ɣ', 'ज़': 'z', 'फ़': 'f',
            'ड़': 'ɽ', 'ढ़': 'ɽʰ',
        }

        self.vowels = {
            'अ': 'ə', 'आ': 'aː', 'इ': 'ɪ', 'ई': 'iː',
            'उ': 'ʊ', 'ऊ': 'uː', 'ए': 'eː', 'ऐ': 'ɛː',
            'ओ': 'oː', 'औ': 'ɔː', 'ऋ': 'ri',
        }

        self.matras = {
            'ा': 'aː', 'ि': 'ɪ', 'ी': 'iː', 'ु': 'ʊ',
            'ू': 'uː', 'े': 'eː', 'ै': 'ɛː', 'ो': 'oː',
            'ौ': 'ɔː', 'ृ': 'ri',
            '्': '',  # Halant - removes inherent vowel
            'ं': 'n',  # Anusvara
            'ँ': '̃',  # Chandrabindu (nasalization)
            'ः': 'h',  # Visarga
        }

    def phonemize(self, text: str) -> str:
        """
        Convert Hindi text to IPA phonemes.

        Args:
            text: Hindi text

        Returns:
            IPA phoneme string
        """
        result = []
        i = 0

        while i < len(text):
            char = text[i]

            # Check for consonant
            if char in self.consonants:
                phoneme = self.consonants[char]

                # Check for following matra or halant
                if i + 1 < len(text) and text[i + 1] in self.matras:
                    matra = text[i + 1]
                    if matra == '्':  # Halant
                        result.append(phoneme)
                    else:
                        result.append(phoneme + self.matras[matra])
                    i += 2
                else:
                    # Add inherent schwa vowel
                    result.append(phoneme + 'ə')
                    i += 1

            # Check for independent vowel
            elif char in self.vowels:
                result.append(self.vowels[char])
                i += 1

            # Check for other matras (shouldn't appear independently)
            elif char in self.matras:
                result.append(self.matras[char])
                i += 1

            # Punctuation and spaces
            elif char in ' \t\n':
                result.append(' ')
                i += 1

            elif char in '।,;:!?.':
                result.append(char if char != '।' else '.')
                i += 1

            else:
                i += 1

        return ''.join(result)


# Factory functions
def create_hindi_processor(config: Optional[HindiProcessorConfig] = None) -> HindiTextProcessor:
    """Create a Hindi text processor."""
    return HindiTextProcessor(config)


def create_hindi_phonemizer() -> HindiPhonemizer:
    """Create a Hindi phonemizer."""
    return HindiPhonemizer()
