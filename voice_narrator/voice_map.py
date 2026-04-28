"""Maps voice/age/mood/language combinations to TTS engine configs."""


# ── Bark speaker presets (English) ──────────────────────────────────

BARK_SPEAKERS = {
    # (voice, age) -> bark preset
    ("man", "adult"):    "v2/en_speaker_6",
    ("man", "young"):    "v2/en_speaker_3",
    ("man", "elderly"):  "v2/en_speaker_0",
    ("grandpa", "elderly"): "v2/en_speaker_0",
    ("grandpa", "adult"):   "v2/en_speaker_0",
    ("woman", "adult"):  "v2/en_speaker_9",
    ("woman", "young"):  "v2/en_speaker_7",
    ("woman", "elderly"): "v2/en_speaker_1",
    ("grandma", "elderly"): "v2/en_speaker_1",
    ("grandma", "adult"):   "v2/en_speaker_1",
    ("kid", "child"):    "v2/en_speaker_4",
    ("kid", "young"):    "v2/en_speaker_4",
}

# Fallbacks for any unmapped combo
BARK_DEFAULT = "v2/en_speaker_6"

# Bark mood tokens prepended/appended to text
BARK_MOOD_TOKENS = {
    "calm":       "",
    "angry":      "[angry] ",
    "happy":      "[laughs] ",
    "sad":        "[sighs] ",
    "excited":    "[laughs] ",
    "scared":     "[gasps] ",
    "whispering": "[whispers] ",
}


# ── Edge TTS voice IDs (Indian languages + English accents) ────────

EDGE_VOICES = {
    # language -> (voice, age) -> edge voice ID
    "hindi": {
        ("man", "adult"):    "hi-IN-MadhurNeural",
        ("man", "young"):    "hi-IN-MadhurNeural",
        ("man", "elderly"):  "hi-IN-MadhurNeural",
        ("grandpa", "elderly"): "hi-IN-MadhurNeural",
        ("grandpa", "adult"):   "hi-IN-MadhurNeural",
        ("woman", "adult"):  "hi-IN-SwaraNeural",
        ("woman", "young"):  "hi-IN-SwaraNeural",
        ("woman", "elderly"): "hi-IN-SwaraNeural",
        ("grandma", "elderly"): "hi-IN-SwaraNeural",
        ("grandma", "adult"):   "hi-IN-SwaraNeural",
        ("kid", "child"):    "hi-IN-SwaraNeural",
        ("kid", "young"):    "hi-IN-SwaraNeural",
    },
    "kannada": {
        ("man", "adult"):    "kn-IN-GaganNeural",
        ("man", "young"):    "kn-IN-GaganNeural",
        ("man", "elderly"):  "kn-IN-GaganNeural",
        ("grandpa", "elderly"): "kn-IN-GaganNeural",
        ("grandpa", "adult"):   "kn-IN-GaganNeural",
        ("woman", "adult"):  "kn-IN-SapnaNeural",
        ("woman", "young"):  "kn-IN-SapnaNeural",
        ("woman", "elderly"): "kn-IN-SapnaNeural",
        ("grandma", "elderly"): "kn-IN-SapnaNeural",
        ("grandma", "adult"):   "kn-IN-SapnaNeural",
        ("kid", "child"):    "kn-IN-SapnaNeural",
        ("kid", "young"):    "kn-IN-SapnaNeural",
    },
    "tamil": {
        ("man", "adult"):    "ta-IN-ValluvarNeural",
        ("man", "young"):    "ta-IN-ValluvarNeural",
        ("man", "elderly"):  "ta-IN-ValluvarNeural",
        ("grandpa", "elderly"): "ta-IN-ValluvarNeural",
        ("grandpa", "adult"):   "ta-IN-ValluvarNeural",
        ("woman", "adult"):  "ta-IN-PallaviNeural",
        ("woman", "young"):  "ta-IN-PallaviNeural",
        ("woman", "elderly"): "ta-IN-PallaviNeural",
        ("grandma", "elderly"): "ta-IN-PallaviNeural",
        ("grandma", "adult"):   "ta-IN-PallaviNeural",
        ("kid", "child"):    "ta-IN-PallaviNeural",
        ("kid", "young"):    "ta-IN-PallaviNeural",
    },
    "telugu": {
        ("man", "adult"):    "te-IN-MohanNeural",
        ("man", "young"):    "te-IN-MohanNeural",
        ("man", "elderly"):  "te-IN-MohanNeural",
        ("grandpa", "elderly"): "te-IN-MohanNeural",
        ("grandpa", "adult"):   "te-IN-MohanNeural",
        ("woman", "adult"):  "te-IN-ShrutiNeural",
        ("woman", "young"):  "te-IN-ShrutiNeural",
        ("woman", "elderly"): "te-IN-ShrutiNeural",
        ("grandma", "elderly"): "te-IN-ShrutiNeural",
        ("grandma", "adult"):   "te-IN-ShrutiNeural",
        ("kid", "child"):    "te-IN-ShrutiNeural",
        ("kid", "young"):    "te-IN-ShrutiNeural",
    },
    "bengali": {
        ("man", "adult"):    "bn-IN-BashkarNeural",
        ("man", "young"):    "bn-IN-BashkarNeural",
        ("man", "elderly"):  "bn-IN-BashkarNeural",
        ("grandpa", "elderly"): "bn-IN-BashkarNeural",
        ("grandpa", "adult"):   "bn-IN-BashkarNeural",
        ("woman", "adult"):  "bn-IN-TanishaaNeural",
        ("woman", "young"):  "bn-IN-TanishaaNeural",
        ("woman", "elderly"): "bn-IN-TanishaaNeural",
        ("grandma", "elderly"): "bn-IN-TanishaaNeural",
        ("grandma", "adult"):   "bn-IN-TanishaaNeural",
        ("kid", "child"):    "bn-IN-TanishaaNeural",
        ("kid", "young"):    "bn-IN-TanishaaNeural",
    },
    "marathi": {
        ("man", "adult"):    "mr-IN-ManoharNeural",
        ("man", "young"):    "mr-IN-ManoharNeural",
        ("man", "elderly"):  "mr-IN-ManoharNeural",
        ("grandpa", "elderly"): "mr-IN-ManoharNeural",
        ("grandpa", "adult"):   "mr-IN-ManoharNeural",
        ("woman", "adult"):  "mr-IN-AarohiNeural",
        ("woman", "young"):  "mr-IN-AarohiNeural",
        ("woman", "elderly"): "mr-IN-AarohiNeural",
        ("grandma", "elderly"): "mr-IN-AarohiNeural",
        ("grandma", "adult"):   "mr-IN-AarohiNeural",
        ("kid", "child"):    "mr-IN-AarohiNeural",
        ("kid", "young"):    "mr-IN-AarohiNeural",
    },
    # English accents via Edge TTS (fallback when Bark unavailable)
    "english_us": {
        ("man", "adult"):    "en-US-GuyNeural",
        ("man", "young"):    "en-US-AndrewNeural",
        ("man", "elderly"):  "en-US-GuyNeural",
        ("grandpa", "elderly"): "en-US-GuyNeural",
        ("grandpa", "adult"):   "en-US-GuyNeural",
        ("woman", "adult"):  "en-US-AriaNeural",
        ("woman", "young"):  "en-US-JennyNeural",
        ("woman", "elderly"): "en-US-AriaNeural",
        ("grandma", "elderly"): "en-US-AriaNeural",
        ("grandma", "adult"):   "en-US-AriaNeural",
        ("kid", "child"):    "en-US-AnaNeural",
        ("kid", "young"):    "en-US-AnaNeural",
    },
    "english_british": {
        ("man", "adult"):    "en-GB-RyanNeural",
        ("man", "young"):    "en-GB-RyanNeural",
        ("man", "elderly"):  "en-GB-RyanNeural",
        ("grandpa", "elderly"): "en-GB-RyanNeural",
        ("grandpa", "adult"):   "en-GB-RyanNeural",
        ("woman", "adult"):  "en-GB-SoniaNeural",
        ("woman", "young"):  "en-GB-LibbyNeural",
        ("woman", "elderly"): "en-GB-SoniaNeural",
        ("grandma", "elderly"): "en-GB-SoniaNeural",
        ("grandma", "adult"):   "en-GB-SoniaNeural",
        ("kid", "child"):    "en-GB-LibbyNeural",
        ("kid", "young"):    "en-GB-LibbyNeural",
    },
    "english_indian": {
        ("man", "adult"):    "en-IN-PrabhatNeural",
        ("man", "young"):    "en-IN-PrabhatNeural",
        ("man", "elderly"):  "en-IN-PrabhatNeural",
        ("grandpa", "elderly"): "en-IN-PrabhatNeural",
        ("grandpa", "adult"):   "en-IN-PrabhatNeural",
        ("woman", "adult"):  "en-IN-NeerjaNeural",
        ("woman", "young"):  "en-IN-NeerjaNeural",
        ("woman", "elderly"): "en-IN-NeerjaNeural",
        ("grandma", "elderly"): "en-IN-NeerjaNeural",
        ("grandma", "adult"):   "en-IN-NeerjaNeural",
        ("kid", "child"):    "en-IN-NeerjaNeural",
        ("kid", "young"):    "en-IN-NeerjaNeural",
    },
    "english_australian": {
        ("man", "adult"):    "en-AU-WilliamMultilingualNeural",
        ("man", "young"):    "en-AU-WilliamMultilingualNeural",
        ("man", "elderly"):  "en-AU-WilliamMultilingualNeural",
        ("grandpa", "elderly"): "en-AU-WilliamMultilingualNeural",
        ("grandpa", "adult"):   "en-AU-WilliamMultilingualNeural",
        ("woman", "adult"):  "en-AU-NatashaNeural",
        ("woman", "young"):  "en-AU-NatashaNeural",
        ("woman", "elderly"): "en-AU-NatashaNeural",
        ("grandma", "elderly"): "en-AU-NatashaNeural",
        ("grandma", "adult"):   "en-AU-NatashaNeural",
        ("kid", "child"):    "en-AU-NatashaNeural",
        ("kid", "young"):    "en-AU-NatashaNeural",
    },
    "english_irish": {
        ("man", "adult"):    "en-IE-ConnorNeural",
        ("man", "young"):    "en-IE-ConnorNeural",
        ("man", "elderly"):  "en-IE-ConnorNeural",
        ("grandpa", "elderly"): "en-IE-ConnorNeural",
        ("grandpa", "adult"):   "en-IE-ConnorNeural",
        ("woman", "adult"):  "en-IE-EmilyNeural",
        ("woman", "young"):  "en-IE-EmilyNeural",
        ("woman", "elderly"): "en-IE-EmilyNeural",
        ("grandma", "elderly"): "en-IE-EmilyNeural",
        ("grandma", "adult"):   "en-IE-EmilyNeural",
        ("kid", "child"):    "en-IE-EmilyNeural",
        ("kid", "young"):    "en-IE-EmilyNeural",
    },
}

# Edge TTS mood -> SSML style mapping (only some voices support these)
EDGE_MOOD_STYLES = {
    "calm":       "calm",
    "angry":      "angry",
    "happy":      "cheerful",
    "sad":        "sad",
    "excited":    "excited",
    "scared":     "fearful",
    "whispering": "whispering",
}

# Edge TTS mood -> rate/pitch tweaks (always works, even without style support)
EDGE_MOOD_PROSODY = {
    "calm":       {"rate": "-5%",  "pitch": "-2Hz"},
    "angry":      {"rate": "+15%", "pitch": "+5Hz"},
    "happy":      {"rate": "+10%", "pitch": "+3Hz"},
    "sad":        {"rate": "-15%", "pitch": "-5Hz"},
    "excited":    {"rate": "+20%", "pitch": "+5Hz"},
    "scared":     {"rate": "+10%", "pitch": "+8Hz"},
    "whispering": {"rate": "-10%", "pitch": "-3Hz"},
}


def is_english(language: str) -> bool:
    return language.startswith("english_")


def get_bark_preset(voice: str, age: str) -> str:
    return BARK_SPEAKERS.get((voice, age), BARK_DEFAULT)


def get_bark_mood_token(mood: str) -> str:
    return BARK_MOOD_TOKENS.get(mood, "")


def get_edge_voice(language: str, voice: str, age: str) -> str:
    lang_voices = EDGE_VOICES.get(language, EDGE_VOICES["english_us"])
    return lang_voices.get((voice, age), list(lang_voices.values())[0])


def get_edge_prosody(mood: str) -> dict:
    return EDGE_MOOD_PROSODY.get(mood, {})
