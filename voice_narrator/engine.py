"""TTS engine using Edge TTS with prosody-based emotions."""

import asyncio
import os
import tempfile
import time

import edge_tts

from voice_narrator.voice_map import get_edge_voice


# ── Mood prosody configs ───────────────────────────────────────────

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


def _get_prosody(mood, voice_type):
    mood_cfg = MOOD_PROSODY.get(mood, MOOD_PROSODY["calm"])
    voice_adj = VOICE_ADJUSTMENTS.get(voice_type, {"rate_add": 0, "pitch_add": 0, "volume_add": 0})
    rate = _parse_val(mood_cfg["rate"], "%") + voice_adj["rate_add"]
    pitch = _parse_val(mood_cfg["pitch"], "Hz") + voice_adj["pitch_add"]
    volume = _parse_val(mood_cfg["volume"], "%") + voice_adj["volume_add"]
    return _fmt_val(rate, "%"), _fmt_val(pitch, "Hz"), _fmt_val(volume, "%")


# ── Synthesis ──────────────────────────────────────────────────────

async def _synthesize_async(text, language, voice, age, mood):
    voice_id = get_edge_voice(language, voice, age)
    rate_str, pitch_str, volume_str = _get_prosody(mood, voice)
    communicate = edge_tts.Communicate(
        text=text, voice=voice_id,
        rate=rate_str, pitch=pitch_str, volume=volume_str,
    )
    path = os.path.join(tempfile.gettempdir(), f"edge_{int(time.time() * 1000)}.mp3")
    await communicate.save(path)
    return path


def synthesize_scene(text, language, voice, age, mood):
    """Synthesize a single scene. Returns path to audio file."""
    return asyncio.run(_synthesize_async(text, language, voice, age, mood))


def get_engine_status():
    return {"edge_tts": True}
