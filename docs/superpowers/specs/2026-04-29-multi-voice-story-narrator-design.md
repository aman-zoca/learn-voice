# Multi-Voice Story Narrator -- Design Spec

## Purpose

A Gradio-based app that takes JSON story scripts and narrates them using multiple voices, languages, moods, and speaker types. Designed for producing YouTube voiceover content. Runs locally with no paid API keys.

## TTS Engine: Hybrid Approach

| Language Category | Engine | Reason |
|-------------------|--------|--------|
| English (US, British, Indian, Australian, Irish) | Bark (Suno AI) | Expressive local TTS with emotion support |
| Indian languages (Hindi, Kannada, Tamil, Telugu, Bengali, Marathi) | Edge TTS | Best free Indian language coverage, no API key |

The engine layer auto-selects Bark or Edge TTS based on the story's language field.

### Bark Details
- Uses voice presets mapped to speaker types (man, woman, kid, grandpa, grandma)
- Mood conditioning via Bark's text prompt tokens: `[laughs]`, `[sighs]`, `[angry]`, `[whispers]`, etc.
- English accent variation via different speaker presets

### Edge TTS Details
- Uses Microsoft Edge's free TTS voices (no API key, requires internet)
- Voice selection by language + gender + age
- SSML style tags for mood where supported (e.g., `cheerful`, `angry`, `sad`)

## JSON Input Schema

```json
{
  "title": "Story Title",
  "language": "hindi",
  "scenes": [
    {
      "speaker": "narrator",
      "voice": "man",
      "age": "adult",
      "mood": "calm",
      "text": "The actual dialogue or narration text..."
    }
  ]
}
```

### Fields

- **title** (string, required): Story name for display
- **language** (string, required): One of `english_us`, `english_british`, `english_indian`, `english_australian`, `english_irish`, `hindi`, `kannada`, `tamil`, `telugu`, `bengali`, `marathi`
- **scenes** (array, required): Ordered list of dialogue entries
  - **speaker** (string, required): Character name (for display)
  - **voice** (string, required): One of `man`, `woman`, `kid`, `grandma`, `grandpa`
  - **age** (string, required): One of `child`, `young`, `adult`, `elderly`
  - **mood** (string, required): One of `calm`, `angry`, `happy`, `sad`, `excited`, `scared`, `whispering`
  - **text** (string, required): The dialogue text in the story's language

## Voice Mapping

### English (Bark)

| Voice + Age | Bark Preset |
|-------------|-------------|
| man / adult | `v2/en_speaker_6` |
| man / young | `v2/en_speaker_3` |
| man / elderly (grandpa) | `v2/en_speaker_0` |
| woman / adult | `v2/en_speaker_9` |
| woman / young | `v2/en_speaker_7` |
| woman / elderly (grandma) | `v2/en_speaker_1` |
| kid / child | `v2/en_speaker_4` |

Accent variants use different Bark speaker presets per English sub-language.

### Indian Languages (Edge TTS)

Each language has male, female, and (where available) child voices mapped to Edge TTS voice IDs. Examples:
- Hindi male: `hi-IN-MadhurNeural`
- Hindi female: `hi-IN-SwaraNeural`
- Kannada male: `kn-IN-GaganNeural`
- Kannada female: `kn-IN-SapnaNeural`

## UI Layout (Gradio)

### Left Panel
- **Example dropdown**: Select from 4 preloaded stories
- **JSON editor**: Large text area to paste/edit JSON scripts
- **Validate button**: Checks JSON structure before playback

### Right Panel
- **Scene list**: Shows each scene with speaker name, mood badge, and individual play button
- **Play All button**: Plays scenes sequentially with short pauses between them
- **Export Full Audio button**: Concatenates all scenes into a single WAV/MP3 file for download

### Bottom
- Progress bar showing current scene during playback
- Status messages

## Dummy Stories (4)

1. **Hindi Family Drama** (`hindi_family.json`) -- Angry grandpa, scared grandchild, calm narrator. A story about a mango tree dispute.
2. **Kannada Folk Tale** (`kannada_folk.json`) -- Village narrator, excited kid, wise grandma. A folk tale about a clever fox.
3. **English Office Comedy** (`english_office.json`, US accent) -- Man and woman coworkers, happy/angry moods. A story about a missing lunch.
4. **English Indian School Story** (`english_indian_school.json`, Indian accent) -- Kid speaker, adult teacher, calm/excited moods. A story about a science fair.

## File Structure

```
voice_narrator/
├── app.py              # Gradio UI
├── engine.py           # Hybrid TTS engine (Bark + Edge TTS)
├── voice_map.py        # Voice/mood/language -> engine config mapping
├── story_schema.py     # JSON validation with Pydantic
├── examples/           # 4 dummy JSON stories
│   ├── hindi_family.json
│   ├── kannada_folk.json
│   ├── english_office.json
│   └── english_indian_school.json
└── output/             # Generated audio files
```

## Dependencies (New)

- `bark` -- Suno Bark TTS
- `edge-tts` -- Microsoft Edge TTS (free, async)
- `pydub` -- Audio concatenation for export
- `gradio` -- Already in project

## Out of Scope

- Real-time voice cloning
- Training custom voices
- Video generation
- Background music/sound effects
