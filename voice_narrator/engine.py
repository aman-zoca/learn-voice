"""Multi-engine TTS: Edge TTS, gTTS, Piper, Silero -- switchable at runtime."""

import asyncio
import os
import tempfile
import time

from voice_narrator.voice_map import (
    get_bark_mood_token,
    get_bark_preset,
    get_edge_voice,
    is_english,
)

# ── Engine availability ────────────────────────────────────────────

ENGINES = {}

try:
    import edge_tts
    ENGINES["Edge TTS"] = True
except ImportError:
    ENGINES["Edge TTS"] = False

try:
    from gtts import gTTS as _gTTS
    ENGINES["gTTS"] = True
except ImportError:
    ENGINES["gTTS"] = False

try:
    from piper import PiperVoice
    ENGINES["Piper"] = True
except ImportError:
    ENGINES["Piper"] = False

try:
    import torch
    ENGINES["Silero"] = True
except ImportError:
    ENGINES["Silero"] = False


def get_available_engines() -> list[str]:
    return [name for name, available in ENGINES.items() if available]


def get_engine_status() -> dict:
    return ENGINES.copy()


# ── Mood prosody configs (for Edge TTS) ────────────────────────────

MOOD_PROSODY = {
    "calm":       {"rate": "-10%", "pitch": "-3Hz",  "volume": "+0%"},
    "angry":      {"rate": "+18%", "pitch": "+8Hz",  "volume": "+20%"},
    "happy":      {"rate": "+12%", "pitch": "+5Hz",  "volume": "+10%"},
    "sad":        {"rate": "-25%", "pitch": "-8Hz",  "volume": "-15%"},
    "excited":    {"rate": "+22%", "pitch": "+8Hz",  "volume": "+15%"},
    "scared":     {"rate": "+8%",  "pitch": "+12Hz", "volume": "-10%"},
    "whispering": {"rate": "-15%", "pitch": "-5Hz",  "volume": "-30%"},
}

VOICE_ADJUSTMENTS = {
    "kid":     {"rate_add": 12, "pitch_add": 10, "volume_add": 5},
    "grandpa": {"rate_add": -15, "pitch_add": -8, "volume_add": -5},
    "grandma": {"rate_add": -10, "pitch_add": -3, "volume_add": -5},
    "man":     {"rate_add": 0,   "pitch_add": 0,  "volume_add": 0},
    "woman":   {"rate_add": 0,   "pitch_add": 0,  "volume_add": 0},
}


def _parse_val(s, suffix):
    return int(s.replace(suffix, "").replace("+", ""))


def _fmt_val(val, suffix):
    return f"+{val}{suffix}" if val >= 0 else f"{val}{suffix}"


def _get_edge_prosody(mood, voice_type):
    mood_cfg = MOOD_PROSODY.get(mood, MOOD_PROSODY["calm"])
    voice_adj = VOICE_ADJUSTMENTS.get(voice_type, {"rate_add": 0, "pitch_add": 0, "volume_add": 0})
    rate = _parse_val(mood_cfg["rate"], "%") + voice_adj["rate_add"]
    pitch = _parse_val(mood_cfg["pitch"], "Hz") + voice_adj["pitch_add"]
    volume = _parse_val(mood_cfg["volume"], "%") + voice_adj["volume_add"]
    return _fmt_val(rate, "%"), _fmt_val(pitch, "Hz"), _fmt_val(volume, "%")


# ── gTTS configs ───────────────────────────────────────────────────

GTTS_LANG_MAP = {
    "english_us": ("en", "com"),
    "english_british": ("en", "co.uk"),
    "english_indian": ("en", "co.in"),
    "english_australian": ("en", "com.au"),
    "english_irish": ("en", "ie"),
    "hindi": ("hi", "com"),
    "kannada": ("kn", "com"),
    "tamil": ("ta", "com"),
    "telugu": ("te", "com"),
    "bengali": ("bn", "com"),
    "marathi": ("mr", "com"),
}

GTTS_MOOD_SLOW = {"sad", "calm", "whispering"}


# ── Piper configs ──────────────────────────────────────────────────

PIPER_MODELS = {
    "english_us":         "en_US-lessac-medium",
    "english_british":    "en_GB-alba-medium",
    "english_indian":     "en_US-lessac-medium",
    "english_australian": "en_US-lessac-medium",
    "english_irish":      "en_US-lessac-medium",
    "hindi":              "hi_IN-HiFi_TTS-medium",
}


# ── Silero configs ─────────────────────────────────────────────────

SILERO_LANG_MAP = {
    "english_us": ("en", "v3_en"),
    "english_british": ("en", "v3_en"),
    "english_indian": ("en", "v3_en"),
    "english_australian": ("en", "v3_en"),
    "english_irish": ("en", "v3_en"),
    "hindi": ("indic", "v4_indic"),
    "kannada": ("indic", "v4_indic"),
    "tamil": ("indic", "v4_indic"),
    "telugu": ("indic", "v4_indic"),
    "bengali": ("indic", "v4_indic"),
    "marathi": ("indic", "v4_indic"),
}

SILERO_SPEAKERS = {
    # (language, voice_type) -> speaker_id
    ("en", "man"):     "en_0",
    ("en", "woman"):   "en_1",
    ("en", "kid"):     "en_2",
    ("en", "grandpa"): "en_3",
    ("en", "grandma"): "en_4",
    ("indic", "man"):     "hindi_male",
    ("indic", "woman"):   "hindi_female",
    ("indic", "kid"):     "hindi_female",
    ("indic", "grandpa"): "hindi_male",
    ("indic", "grandma"): "hindi_female",
}

# Kannada-specific speakers
SILERO_INDIC_SPEAKERS = {
    "hindi":   {"man": "hindi_male",   "woman": "hindi_female"},
    "kannada": {"man": "kannada_male", "woman": "kannada_female"},
    "tamil":   {"man": "tamil_male",   "woman": "tamil_female"},
    "telugu":  {"man": "telugu_male",  "woman": "telugu_female"},
    "bengali": {"man": "bengali_male", "woman": "bengali_female"},
    "marathi": {"man": "marathi_male", "woman": "marathi_female"},
}

_silero_models = {}


def _get_silero_model(language):
    lang_cfg = SILERO_LANG_MAP.get(language)
    if not lang_cfg:
        return None, None
    lang, model_id = lang_cfg
    if model_id not in _silero_models:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=lang,
            speaker=model_id,
        )
        _silero_models[model_id] = model
    return _silero_models[model_id], lang


def _get_silero_speaker(language, voice_type):
    lang_cfg = SILERO_LANG_MAP.get(language)
    if not lang_cfg:
        return "en_0"
    lang, _ = lang_cfg
    if lang == "indic":
        lang_speakers = SILERO_INDIC_SPEAKERS.get(language, {})
        # Map voice types to male/female
        if voice_type in ("man", "grandpa"):
            return lang_speakers.get("man", "hindi_male")
        else:
            return lang_speakers.get("woman", "hindi_female")
    return SILERO_SPEAKERS.get((lang, voice_type), "en_0")


# ── Engine implementations ─────────────────────────────────────────

async def _synth_edge(text, language, voice, age, mood):
    voice_id = get_edge_voice(language, voice, age)
    rate_str, pitch_str, volume_str = _get_edge_prosody(mood, voice)
    communicate = edge_tts.Communicate(
        text=text, voice=voice_id,
        rate=rate_str, pitch=pitch_str, volume=volume_str,
    )
    path = os.path.join(tempfile.gettempdir(), f"edge_{int(time.time() * 1000)}.mp3")
    await communicate.save(path)
    return path


def _synth_gtts(text, language, voice, age, mood):
    lang_cfg = GTTS_LANG_MAP.get(language, ("en", "com"))
    lang_code, tld = lang_cfg
    slow = mood in GTTS_MOOD_SLOW
    tts = _gTTS(text=text, lang=lang_code, tld=tld, slow=slow)
    path = os.path.join(tempfile.gettempdir(), f"gtts_{int(time.time() * 1000)}.mp3")
    tts.save(path)
    return path


def _synth_piper(text, language, voice, age, mood):
    model_name = PIPER_MODELS.get(language)
    if not model_name:
        raise RuntimeError(f"Piper: no model for '{language}'. Supported: {list(PIPER_MODELS.keys())}")

    data_dir = os.path.join(os.path.dirname(__file__), ".piper_models")
    os.makedirs(data_dir, exist_ok=True)

    # Look for existing model file
    onnx_path = None
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".onnx") and model_name in f:
                onnx_path = os.path.join(root, f)
                break

    if onnx_path is None:
        raise RuntimeError(
            f"Piper model '{model_name}' not found. Download it:\n"
            f"  mkdir -p {data_dir}\n"
            f"  cd {data_dir}\n"
            f"  Download from: https://huggingface.co/rhasspy/piper-voices/tree/main\n"
            f"  Place the .onnx and .onnx.json files in {data_dir}"
        )

    import wave
    piper_voice = PiperVoice.load(onnx_path)
    path = os.path.join(tempfile.gettempdir(), f"piper_{int(time.time() * 1000)}.wav")
    with wave.open(path, "wb") as wav_file:
        piper_voice.synthesize(text, wav_file)
    return path


def _synth_silero(text, language, voice, age, mood):
    model, lang = _get_silero_model(language)
    if model is None:
        raise RuntimeError(f"Silero: no model for {language}")
    speaker = _get_silero_speaker(language, voice)
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=48000)
    path = os.path.join(tempfile.gettempdir(), f"silero_{int(time.time() * 1000)}.wav")
    import torchaudio
    torchaudio.save(path, audio.unsqueeze(0), 48000)
    return path


# ── Main entry point ───────────────────────────────────────────────

def synthesize_scene(text, language, voice, age, mood, engine="Edge TTS"):
    """Synthesize a scene using the specified engine."""
    if engine == "Edge TTS":
        if not ENGINES.get("Edge TTS"):
            raise RuntimeError("Edge TTS not installed: pip install edge-tts")
        return asyncio.run(_synth_edge(text, language, voice, age, mood))

    elif engine == "gTTS":
        if not ENGINES.get("gTTS"):
            raise RuntimeError("gTTS not installed: pip install gTTS")
        return _synth_gtts(text, language, voice, age, mood)

    elif engine == "Piper":
        if not ENGINES.get("Piper"):
            raise RuntimeError("Piper not installed: pip install piper-tts")
        return _synth_piper(text, language, voice, age, mood)

    elif engine == "Silero":
        if not ENGINES.get("Silero"):
            raise RuntimeError("Silero not installed: pip install silero-tts torch torchaudio")
        return _synth_silero(text, language, voice, age, mood)

    else:
        raise ValueError(f"Unknown engine: {engine}. Available: {get_available_engines()}")
