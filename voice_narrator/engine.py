"""Hybrid TTS engine: Bark for English, Edge TTS for Indian languages.

Uses aggressive rate/pitch/volume prosody to make speech sound emotional and human.
"""

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

# Try to import Bark
BARK_AVAILABLE = False
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    import soundfile as sf
    BARK_AVAILABLE = True
except ImportError:
    pass

# Try to import Edge TTS
EDGE_AVAILABLE = False
try:
    import edge_tts
    EDGE_AVAILABLE = True
except ImportError:
    pass

_bark_loaded = False


# ── Mood prosody configs ───────────────────────────────────────────
# Aggressive rate/pitch/volume to make each mood sound distinct and human

MOOD_PROSODY = {
    "calm":       {"rate": "-10%", "pitch": "-3Hz",  "volume": "+0%"},
    "angry":      {"rate": "+18%", "pitch": "+8Hz",  "volume": "+20%"},
    "happy":      {"rate": "+12%", "pitch": "+5Hz",  "volume": "+10%"},
    "sad":        {"rate": "-25%", "pitch": "-8Hz",  "volume": "-15%"},
    "excited":    {"rate": "+22%", "pitch": "+8Hz",  "volume": "+15%"},
    "scared":     {"rate": "+8%",  "pitch": "+12Hz", "volume": "-10%"},
    "whispering": {"rate": "-15%", "pitch": "-5Hz",  "volume": "-30%"},
}

# Voice-type prosody adjustments for character variety
VOICE_ADJUSTMENTS = {
    "kid":     {"rate_add": 12, "pitch_add": 10, "volume_add": 5},
    "grandpa": {"rate_add": -15, "pitch_add": -8, "volume_add": -5},
    "grandma": {"rate_add": -10, "pitch_add": -3, "volume_add": -5},
    "man":     {"rate_add": 0,   "pitch_add": 0,  "volume_add": 0},
    "woman":   {"rate_add": 0,   "pitch_add": 0,  "volume_add": 0},
}


def _parse_prosody_val(s: str, suffix: str) -> int:
    """Parse '+10%' or '-5Hz' to int."""
    return int(s.replace(suffix, "").replace("+", ""))


def _format_prosody_val(val: int, suffix: str) -> str:
    """Format int to '+10%' or '-5Hz'."""
    return f"+{val}{suffix}" if val >= 0 else f"{val}{suffix}"


def _get_prosody(mood: str, voice_type: str):
    """Combine mood + voice type into final rate/pitch/volume strings."""
    mood_cfg = MOOD_PROSODY.get(mood, MOOD_PROSODY["calm"])
    voice_adj = VOICE_ADJUSTMENTS.get(voice_type, {"rate_add": 0, "pitch_add": 0, "volume_add": 0})

    rate = _parse_prosody_val(mood_cfg["rate"], "%") + voice_adj["rate_add"]
    pitch = _parse_prosody_val(mood_cfg["pitch"], "Hz") + voice_adj["pitch_add"]
    volume = _parse_prosody_val(mood_cfg["volume"], "%") + voice_adj["volume_add"]

    return (
        _format_prosody_val(rate, "%"),
        _format_prosody_val(pitch, "Hz"),
        _format_prosody_val(volume, "%"),
    )


def _ensure_bark():
    global _bark_loaded
    if not _bark_loaded:
        preload_models()
        _bark_loaded = True


def synthesize_bark(text: str, voice: str, age: str, mood: str) -> str:
    """Generate speech using Bark. Returns path to wav file."""
    _ensure_bark()
    preset = get_bark_preset(voice, age)
    mood_token = get_bark_mood_token(mood)
    prompted_text = f"{mood_token}{text}"
    audio_array = generate_audio(prompted_text, history_prompt=preset)
    output_path = os.path.join(tempfile.gettempdir(), f"bark_{int(time.time() * 1000)}.wav")
    sf.write(output_path, audio_array, SAMPLE_RATE)
    return output_path


async def _synthesize_edge_async(text: str, language: str, voice: str, age: str, mood: str) -> str:
    """Generate speech using Edge TTS with prosody for emotional speech."""
    voice_id = get_edge_voice(language, voice, age)
    rate_str, pitch_str, volume_str = _get_prosody(mood, voice)

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice_id,
        rate=rate_str,
        pitch=pitch_str,
        volume=volume_str,
    )
    output_path = os.path.join(tempfile.gettempdir(), f"edge_{int(time.time() * 1000)}.mp3")
    await communicate.save(output_path)
    return output_path


def synthesize_edge(text: str, language: str, voice: str, age: str, mood: str) -> str:
    """Sync wrapper for Edge TTS."""
    return asyncio.run(_synthesize_edge_async(text, language, voice, age, mood))


def synthesize_scene(text: str, language: str, voice: str, age: str, mood: str) -> str:
    """
    Synthesize a single scene. Auto-selects engine:
    - English + Bark available -> Bark
    - Otherwise -> Edge TTS
    Returns path to audio file.
    """
    use_bark = is_english(language) and BARK_AVAILABLE

    if use_bark:
        return synthesize_bark(text, voice, age, mood)
    elif EDGE_AVAILABLE:
        return synthesize_edge(text, language, voice, age, mood)
    else:
        raise RuntimeError("No TTS engine available. Install bark or edge-tts:\n  pip install git+https://github.com/suno-ai/bark.git edge-tts")


def get_engine_status() -> dict:
    return {
        "bark": BARK_AVAILABLE,
        "edge_tts": EDGE_AVAILABLE,
    }
