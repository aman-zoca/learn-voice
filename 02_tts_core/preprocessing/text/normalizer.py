"""
Text Normalizer Module
======================
Normalizes text for TTS by handling numbers, abbreviations, and special characters.
"""

import re
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field

try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False

try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False


@dataclass
class NormalizerConfig:
    """Configuration for text normalization."""
    language: str = 'en'
    lowercase: bool = False
    remove_extra_spaces: bool = True
    expand_numbers: bool = True
    expand_abbreviations: bool = True
    expand_currency: bool = True
    expand_ordinals: bool = True
    expand_time: bool = True
    expand_dates: bool = True


# Common abbreviations for English
ENGLISH_ABBREVIATIONS: Dict[str, str] = {
    'mr.': 'mister',
    'mrs.': 'missus',
    'ms.': 'miss',
    'dr.': 'doctor',
    'prof.': 'professor',
    'jr.': 'junior',
    'sr.': 'senior',
    'st.': 'saint',
    'ave.': 'avenue',
    'blvd.': 'boulevard',
    'rd.': 'road',
    'apt.': 'apartment',
    'etc.': 'etcetera',
    'i.e.': 'that is',
    'e.g.': 'for example',
    'vs.': 'versus',
    'no.': 'number',
    'approx.': 'approximately',
    'govt.': 'government',
    'dept.': 'department',
    'inc.': 'incorporated',
    'ltd.': 'limited',
    'co.': 'company',
    'corp.': 'corporation',
    'ft.': 'feet',
    'in.': 'inches',
    'lb.': 'pounds',
    'oz.': 'ounces',
    'min.': 'minutes',
    'sec.': 'seconds',
    'hr.': 'hour',
    'hrs.': 'hours',
    'km.': 'kilometers',
    'kg.': 'kilograms',
}

# Hindi abbreviations
HINDI_ABBREVIATIONS: Dict[str, str] = {
    'डॉ.': 'डॉक्टर',
    'श्री': 'श्रीमान',
    'श्रीमती': 'श्रीमती',
    'कु.': 'कुमारी',
    'प्रो.': 'प्रोफ़ेसर',
    'इंच': 'इंच',
    'किमी': 'किलोमीटर',
    'किग्रा': 'किलोग्राम',
    'मी': 'मीटर',
    'सेमी': 'सेंटीमीटर',
    'लि.': 'लिमिटेड',
}


class TextNormalizer:
    """
    Text normalizer for TTS preprocessing.

    Converts written text to spoken form by:
    - Expanding numbers to words
    - Expanding abbreviations
    - Normalizing punctuation
    - Handling special characters
    """

    def __init__(self, config: Optional[NormalizerConfig] = None):
        self.config = config or NormalizerConfig()

        # Set up abbreviations based on language
        if self.config.language == 'hi':
            self.abbreviations = HINDI_ABBREVIATIONS
        else:
            self.abbreviations = ENGLISH_ABBREVIATIONS

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Number patterns
        self.number_pattern = re.compile(r'\d+(?:\.\d+)?')
        self.ordinal_pattern = re.compile(r'(\d+)(st|nd|rd|th)\b', re.IGNORECASE)

        # Currency patterns
        self.currency_patterns = {
            '$': ('dollar', 'dollars', 'cent', 'cents'),
            '£': ('pound', 'pounds', 'pence', 'pence'),
            '€': ('euro', 'euros', 'cent', 'cents'),
            '₹': ('rupee', 'rupees', 'paisa', 'paise'),
            '¥': ('yen', 'yen', 'sen', 'sen'),
        }
        self.currency_pattern = re.compile(
            r'([' + ''.join(self.currency_patterns.keys()) + r'])\s*(\d+(?:\.\d+)?)'
        )

        # Time pattern (12:30, 14:45)
        self.time_pattern = re.compile(r'(\d{1,2}):(\d{2})(?:\s*(am|pm|AM|PM))?')

        # Date patterns
        self.date_pattern_mdy = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{2,4})')  # MM/DD/YYYY
        self.date_pattern_dmy = re.compile(r'(\d{1,2})-(\d{1,2})-(\d{2,4})')  # DD-MM-YYYY

        # Whitespace
        self.whitespace_pattern = re.compile(r'\s+')

        # URL pattern
        self.url_pattern = re.compile(
            r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}'
            r'\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        )

        # Email pattern
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

    def normalize(self, text: str) -> str:
        """
        Normalize text through the full pipeline.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Basic cleaning
        text = self._clean_text(text)

        # Handle URLs and emails
        text = self._handle_urls(text)
        text = self._handle_emails(text)

        # Expand abbreviations
        if self.config.expand_abbreviations:
            text = self._expand_abbreviations(text)

        # Expand currency
        if self.config.expand_currency:
            text = self._expand_currency(text)

        # Expand time
        if self.config.expand_time:
            text = self._expand_time(text)

        # Expand dates
        if self.config.expand_dates:
            text = self._expand_dates(text)

        # Expand ordinals (before general numbers)
        if self.config.expand_ordinals:
            text = self._expand_ordinals(text)

        # Expand numbers
        if self.config.expand_numbers:
            text = self._expand_numbers(text)

        # Final cleanup
        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_extra_spaces:
            text = self.whitespace_pattern.sub(' ', text).strip()

        return text

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)

        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-\(\)\[\]\{\}@#\$%\^&\*\/\\₹£€¥]', '', text)

        return text

    def _handle_urls(self, text: str) -> str:
        """Replace URLs with placeholder or expand."""
        return self.url_pattern.sub(' URL ', text)

    def _handle_emails(self, text: str) -> str:
        """Expand email addresses."""
        def expand_email(match):
            email = match.group(0)
            parts = email.replace('@', ' at ').replace('.', ' dot ')
            return parts

        return self.email_pattern.sub(expand_email, text)

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations."""
        for abbrev, expansion in self.abbreviations.items():
            pattern = re.compile(re.escape(abbrev), re.IGNORECASE)
            text = pattern.sub(expansion, text)

        return text

    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to words."""
        def number_to_words(match):
            num_str = match.group(0)

            try:
                num = float(num_str)

                if num.is_integer():
                    num = int(num)

                if NUM2WORDS_AVAILABLE:
                    lang = 'hi' if self.config.language == 'hi' else 'en'
                    return num2words(num, lang=lang)
                else:
                    return self._basic_number_to_words(num)

            except (ValueError, OverflowError):
                return num_str

        return self.number_pattern.sub(number_to_words, text)

    def _basic_number_to_words(self, num) -> str:
        """Basic number to words conversion without num2words."""
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
                'seventy', 'eighty', 'ninety']

        if isinstance(num, float):
            integer_part = int(num)
            decimal_part = str(num).split('.')[1]
            result = self._basic_number_to_words(integer_part)
            result += ' point '
            result += ' '.join(ones[int(d)] for d in decimal_part)
            return result

        if num < 0:
            return 'minus ' + self._basic_number_to_words(-num)

        if num == 0:
            return 'zero'

        if num < 20:
            return ones[num]

        if num < 100:
            return tens[num // 10] + ('' if num % 10 == 0 else ' ' + ones[num % 10])

        if num < 1000:
            return (ones[num // 100] + ' hundred' +
                    ('' if num % 100 == 0 else ' and ' + self._basic_number_to_words(num % 100)))

        if num < 1000000:
            return (self._basic_number_to_words(num // 1000) + ' thousand' +
                    ('' if num % 1000 == 0 else ' ' + self._basic_number_to_words(num % 1000)))

        if num < 1000000000:
            return (self._basic_number_to_words(num // 1000000) + ' million' +
                    ('' if num % 1000000 == 0 else ' ' + self._basic_number_to_words(num % 1000000)))

        return str(num)

    def _expand_ordinals(self, text: str) -> str:
        """Expand ordinal numbers (1st, 2nd, etc.)."""
        def ordinal_to_words(match):
            num = int(match.group(1))

            if NUM2WORDS_AVAILABLE:
                lang = 'hi' if self.config.language == 'hi' else 'en'
                return num2words(num, to='ordinal', lang=lang)
            else:
                return self._basic_ordinal_to_words(num)

        return self.ordinal_pattern.sub(ordinal_to_words, text)

    def _basic_ordinal_to_words(self, num: int) -> str:
        """Basic ordinal to words conversion."""
        ordinals = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
            11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth',
            15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth',
            19: 'nineteenth', 20: 'twentieth', 30: 'thirtieth', 40: 'fortieth',
            50: 'fiftieth', 60: 'sixtieth', 70: 'seventieth', 80: 'eightieth',
            90: 'ninetieth', 100: 'hundredth'
        }

        if num in ordinals:
            return ordinals[num]

        if num < 100:
            tens = num // 10 * 10
            ones = num % 10
            if ones == 0:
                return ordinals.get(tens, str(num) + 'th')
            else:
                base = self._basic_number_to_words(tens)
                return base + ' ' + ordinals.get(ones, str(ones) + 'th')

        return str(num) + 'th'

    def _expand_currency(self, text: str) -> str:
        """Expand currency symbols and amounts."""
        def currency_to_words(match):
            symbol = match.group(1)
            amount = float(match.group(2))

            names = self.currency_patterns.get(symbol, ('unit', 'units', 'cent', 'cents'))

            if amount == 1:
                unit_name = names[0]
            else:
                unit_name = names[1]

            if amount == int(amount):
                num_words = self._basic_number_to_words(int(amount))
            else:
                dollars = int(amount)
                cents = int(round((amount - dollars) * 100))

                if dollars == 1:
                    num_words = f"one {names[0]}"
                else:
                    num_words = f"{self._basic_number_to_words(dollars)} {names[1]}"

                if cents > 0:
                    if cents == 1:
                        num_words += f" and one {names[2]}"
                    else:
                        num_words += f" and {self._basic_number_to_words(cents)} {names[3]}"

                return num_words

            return f"{num_words} {unit_name}"

        return self.currency_pattern.sub(currency_to_words, text)

    def _expand_time(self, text: str) -> str:
        """Expand time expressions."""
        def time_to_words(match):
            hour = int(match.group(1))
            minute = int(match.group(2))
            period = match.group(3)

            result = self._basic_number_to_words(hour)

            if minute == 0:
                result += " o'clock"
            elif minute < 10:
                result += f" oh {self._basic_number_to_words(minute)}"
            else:
                result += f" {self._basic_number_to_words(minute)}"

            if period:
                period = period.lower()
                if period == 'am':
                    result += ' A M'
                else:
                    result += ' P M'

            return result

        return self.time_pattern.sub(time_to_words, text)

    def _expand_dates(self, text: str) -> str:
        """Expand date expressions."""
        months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']

        def date_to_words_mdy(match):
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))

            if year < 100:
                year += 2000 if year < 50 else 1900

            month_name = months[month] if 1 <= month <= 12 else str(month)
            day_words = self._basic_ordinal_to_words(day)
            year_words = self._basic_number_to_words(year)

            return f"{month_name} {day_words}, {year_words}"

        def date_to_words_dmy(match):
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))

            if year < 100:
                year += 2000 if year < 50 else 1900

            month_name = months[month] if 1 <= month <= 12 else str(month)
            day_words = self._basic_ordinal_to_words(day)
            year_words = self._basic_number_to_words(year)

            return f"the {day_words} of {month_name}, {year_words}"

        text = self.date_pattern_mdy.sub(date_to_words_mdy, text)
        text = self.date_pattern_dmy.sub(date_to_words_dmy, text)

        return text


# Factory function
def create_normalizer(language: str = 'en', **kwargs) -> TextNormalizer:
    """Create a text normalizer for the specified language."""
    config = NormalizerConfig(language=language, **kwargs)
    return TextNormalizer(config)
