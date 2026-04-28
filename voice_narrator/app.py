"""Multi-Voice Story Narrator -- Gradio UI."""

import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
from pydub import AudioSegment

from voice_narrator.engine import get_available_engines, get_engine_status, synthesize_scene
from voice_narrator.story_schema import Story

EXAMPLES_DIR = Path(__file__).parent / "examples"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MOOD_BADGES = {
    "calm": "😌 calm",
    "angry": "😡 angry",
    "happy": "😄 happy",
    "sad": "😢 sad",
    "excited": "🤩 excited",
    "scared": "😨 scared",
    "whispering": "🤫 whispering",
}

VOICE_ICONS = {
    "man": "👨",
    "woman": "👩",
    "kid": "👦",
    "grandma": "👵",
    "grandpa": "👴",
}


def load_example(example_name: str) -> str:
    """Load an example JSON story."""
    file_map = {
        "Hindi - Family Drama (आम के पेड़ की लड़ाई)": "hindi_family.json",
        "Kannada - Folk Tale (ಬುದ್ಧಿವಂತ ನರಿ)": "kannada_folk.json",
        "English US - Office Comedy (The Missing Lunch)": "english_office.json",
        "English Indian - School Story (The Science Fair)": "english_indian_school.json",
    }
    filename = file_map.get(example_name)
    if filename:
        filepath = EXAMPLES_DIR / filename
        return filepath.read_text(encoding="utf-8")
    return ""


def validate_json(json_text: str) -> tuple[str, str]:
    """Validate JSON against our story schema. Returns (status, formatted_preview)."""
    if not json_text or not json_text.strip():
        return "⚠️ Please paste a JSON story.", ""

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}", ""

    try:
        story = Story(**data)
    except Exception as e:
        return f"❌ Schema error: {e}", ""

    # Build a nice preview
    lines = [f"**{story.title}** ({story.language})", f"Scenes: {len(story.scenes)}", ""]
    for i, scene in enumerate(story.scenes, 1):
        icon = VOICE_ICONS.get(scene.voice, "🗣️")
        mood = MOOD_BADGES.get(scene.mood, scene.mood)
        text_preview = scene.text[:60] + ("..." if len(scene.text) > 60 else "")
        lines.append(f"{i}. {icon} **{scene.speaker}** [{mood}]: {text_preview}")

    return "✅ Valid story!", "\n".join(lines)


def generate_scene_audio(json_text: str, scene_index: int, engine: str = "Edge TTS", progress=gr.Progress()):
    """Generate audio for a single scene."""
    try:
        data = json.loads(json_text)
        story = Story(**data)
    except Exception as e:
        return None, f"❌ Error: {e}"

    if scene_index < 0 or scene_index >= len(story.scenes):
        return None, f"❌ Invalid scene index: {scene_index}"

    scene = story.scenes[scene_index]
    progress(0.3, desc=f"[{engine}] Generating: {scene.speaker}...")

    try:
        audio_path = synthesize_scene(
            text=scene.text,
            language=story.language,
            voice=scene.voice,
            age=scene.age,
            mood=scene.mood,
            engine=engine,
        )
        progress(1.0, desc="Done!")
        icon = VOICE_ICONS.get(scene.voice, "🗣️")
        mood = MOOD_BADGES.get(scene.mood, scene.mood)
        return audio_path, f"✅ [{engine}] {icon} {scene.speaker} [{mood}]"
    except Exception as e:
        return None, f"❌ [{engine}] Failed: {e}"


MOOD_COLORS = {
    "calm": "#e3f2fd",
    "angry": "#ffcdd2",
    "happy": "#fff9c4",
    "sad": "#e1bee7",
    "excited": "#ffe0b2",
    "scared": "#d7ccc8",
    "whispering": "#f3e5f5",
}


def _build_lyrics_html(story, scene_timestamps):
    """Build HTML lyrics with data-start/data-end for JS highlighting."""
    html = f"""
    <div id="lyrics-container" style="padding:10px;">
        <h2 style="text-align:center; margin-bottom:4px;">{story.title}</h2>
        <p style="text-align:center; color:#888; margin-bottom:16px;"><em>{story.language}</em></p>
        <hr style="margin-bottom:16px;">
    """
    for i, scene in enumerate(story.scenes):
        icon = VOICE_ICONS.get(scene.voice, "🗣️")
        mood = MOOD_BADGES.get(scene.mood, scene.mood)
        bg = MOOD_COLORS.get(scene.mood, "#f5f5f5")
        start_s = scene_timestamps[i]["start"] / 1000.0
        end_s = scene_timestamps[i]["end"] / 1000.0

        html += f"""
        <div class="lyric-line" id="lyric-{i}"
             data-start="{start_s:.2f}" data-end="{end_s:.2f}"
             style="padding:12px 16px; margin-bottom:8px; border-radius:8px;
                    border-left:4px solid transparent;
                    transition: all 0.3s ease; opacity:0.55;">
            <div style="font-weight:bold; font-size:15px; margin-bottom:4px; color:#f0f0f0;">
                {icon} <span style="color:#64b5f6;">{scene.speaker}</span> <span style="font-size:12px; color:#bbb;">[{mood}]</span>
            </div>
            <div style="font-size:15px; line-height:1.6; color:#e0e0e0;">
                {scene.text}
            </div>
        </div>
        """

    html += "</div>"
    # Reset the flag so the script re-attaches on each new generation
    html += "<script>window._lyricsSyncAttached = false;</script>"
    html += LYRICS_SYNC_JS_SNIPPET
    return html


def generate_all_audio(json_text: str, engine: str = "Edge TTS", progress=gr.Progress()):
    """Generate audio for all scenes and concatenate into one file.
    Returns (audio_path, status_log, lyrics_html).
    """
    try:
        data = json.loads(json_text)
        story = Story(**data)
    except Exception as e:
        return None, f"❌ Error: {e}", ""

    audio_files = []
    scene_durations = []
    total = len(story.scenes)
    status_lines = [f"Engine: **{engine}**", ""]

    for i, scene in enumerate(story.scenes):
        icon = VOICE_ICONS.get(scene.voice, "🗣️")
        mood = MOOD_BADGES.get(scene.mood, scene.mood)
        progress((i + 0.5) / total, desc=f"[{engine}] Scene {i+1}/{total}: {scene.speaker}...")

        try:
            audio_path = synthesize_scene(
                text=scene.text,
                language=story.language,
                voice=scene.voice,
                age=scene.age,
                mood=scene.mood,
                engine=engine,
            )
            audio_files.append(audio_path)
            seg = AudioSegment.from_file(audio_path)
            scene_durations.append(len(seg))
            status_lines.append(f"✅ {i+1}. {icon} {scene.speaker} [{mood}]")
        except Exception as e:
            status_lines.append(f"❌ {i+1}. {icon} {scene.speaker} - Failed: {e}")
            scene_durations.append(0)

        progress((i + 1) / total, desc=f"Done {i+1}/{total}")

    if not audio_files:
        return None, "❌ No audio generated.", ""

    # Compute timestamps (ms) for each scene
    silence_ms = 800
    scene_timestamps = []
    cursor = 0
    for i, dur in enumerate(scene_durations):
        if i > 0:
            cursor += silence_ms
        scene_timestamps.append({"start": cursor, "end": cursor + dur})
        cursor += dur

    # Concatenate all audio with pause between scenes
    progress(0.95, desc="Merging audio...")
    silence = AudioSegment.silent(duration=silence_ms)
    combined = AudioSegment.empty()

    for path in audio_files:
        segment = AudioSegment.from_file(path)
        if len(combined) > 0:
            combined += silence
        combined += segment

    output_path = str(OUTPUT_DIR / f"{story.title[:30]}_{int(time.time())}.mp3")
    combined.export(output_path, format="mp3")

    status = "\n".join(status_lines)
    status += f"\n\n🎵 Full story exported: {os.path.basename(output_path)}"
    lyrics_html = _build_lyrics_html(story, scene_timestamps)
    return output_path, status, lyrics_html


def get_scene_choices(json_text: str) -> list[str]:
    """Parse JSON and return scene labels for the dropdown."""
    try:
        data = json.loads(json_text)
        story = Story(**data)
        choices = []
        for i, scene in enumerate(story.scenes):
            icon = VOICE_ICONS.get(scene.voice, "🗣️")
            choices.append(f"{i}: {icon} {scene.speaker} ({scene.mood}) - {scene.text[:40]}...")
        return choices
    except Exception:
        return []


def update_scene_dropdown(json_text: str):
    """Update scene dropdown when JSON changes."""
    choices = get_scene_choices(json_text)
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


# ── Build the UI ────────────────────────────────────────────────────

available_engines = get_available_engines()
engine_text = ", ".join(available_engines) if available_engines else "No engines available!"

ENGINE_INFO = {
    "Edge TTS": "Microsoft voices, best emotions via prosody, all Indian langs. Needs internet.",
    "gTTS": "Google voices, simple & reliable, all Indian langs. Needs internet.",
    "Piper": "Super fast, fully offline, English + Hindi only.",
    "Silero": "Offline, Hindi + Kannada + more Indian langs, male/female voices.",
}


APP_CSS = """
    .scene-preview { font-size: 14px; line-height: 1.8; }
    .lyric-active {
        opacity: 1 !important;
        background: rgba(76,175,80,0.15) !important;
        border-left-color: #4caf50 !important;
        transform: scale(1.02);
        box-shadow: 0 2px 12px rgba(76,175,80,0.3);
    }
    .lyric-active div:first-child span:first-of-type {
        color: #69f0ae !important;
    }
    .lyric-done {
        opacity: 0.35 !important;
    }
"""

LYRICS_SYNC_JS_SNIPPET = """
<script>
(function() {
    if (window._lyricsSyncAttached) return;
    window._lyricsSyncAttached = true;

    function findAudio() {
        // Try multiple selectors for different Gradio versions
        return document.querySelector('#full-story-audio audio')
            || document.querySelector('#full-story-audio source')?.parentElement
            || document.querySelector('audio');
    }

    function highlight(t) {
        var lines = document.querySelectorAll('.lyric-line');
        for (var i = 0; i < lines.length; i++) {
            var el = lines[i];
            var start = parseFloat(el.getAttribute('data-start'));
            var end = parseFloat(el.getAttribute('data-end'));
            el.classList.remove('lyric-active', 'lyric-done');
            if (t >= start && t < end) {
                el.classList.add('lyric-active');
                el.scrollIntoView({behavior: 'smooth', block: 'center'});
            } else if (t >= end) {
                el.classList.add('lyric-done');
            }
        }
    }

    var poll = setInterval(function() {
        var audioEl = findAudio();
        if (!audioEl) return;
        clearInterval(poll);
        audioEl.addEventListener('timeupdate', function() { highlight(audioEl.currentTime); });
        audioEl.addEventListener('seeked', function() { highlight(audioEl.currentTime); });
        audioEl.addEventListener('ended', function() {
            document.querySelectorAll('.lyric-line').forEach(function(el) {
                el.classList.remove('lyric-active');
                el.classList.add('lyric-done');
            });
        });
    }, 300);
})();
</script>
"""

with gr.Blocks(title="Multi-Voice Story Narrator") as app:

    gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <h1>Multi-Voice Story Narrator</h1>
            <p>Paste a JSON story script and hear it narrated with multiple voices, moods, and languages.</p>
            <p style="color: #666; font-size: 13px;">Engines: {engine_text}</p>
        </div>
    """)

    # ── Engine Selector ──────────────────────────────────────────────
    with gr.Row():
        engine_radio = gr.Radio(
            choices=available_engines,
            value=available_engines[0] if available_engines else "Edge TTS",
            label="TTS Engine",
            info="Switch engines to compare voices. Each engine sounds different!",
        )
        engine_info = gr.Textbox(
            value=ENGINE_INFO.get(available_engines[0], "") if available_engines else "",
            label="Engine Info",
            interactive=False,
            lines=1,
        )

    def _update_engine_info(engine_name):
        return ENGINE_INFO.get(engine_name, "")

    engine_radio.change(fn=_update_engine_info, inputs=[engine_radio], outputs=[engine_info])

    with gr.Row():
        # ── Left Panel: JSON Editor ────────────────────────────────
        with gr.Column(scale=1):
            example_dropdown = gr.Dropdown(
                choices=[
                    "Hindi - Family Drama (आम के पेड़ की लड़ाई)",
                    "Kannada - Folk Tale (ಬುದ್ಧಿವಂತ ನರಿ)",
                    "English US - Office Comedy (The Missing Lunch)",
                    "English Indian - School Story (The Science Fair)",
                ],
                label="Load Example Story",
                value=None,
            )

            json_input = gr.Code(
                label="Story JSON",
                language="json",
                lines=20,
            )

            with gr.Row():
                validate_btn = gr.Button("Validate JSON", variant="secondary")
                copy_prompt_btn = gr.Button("Copy AI Prompt", variant="secondary")

            prompt_output = gr.Textbox(
                label="AI Prompt (select all + copy, or click Copy below)",
                lines=8,
                visible=False,
            )
            copy_to_clipboard_btn = gr.Button("Copy to Clipboard", visible=False, size="sm")

        # ── Right Panel: Playback ──────────────────────────────────
        with gr.Column(scale=1):
            validation_status = gr.Textbox(label="Status", interactive=False)
            scene_preview = gr.Markdown(label="Scene Preview", elem_classes=["scene-preview"])

            gr.HTML("<hr>")
            gr.HTML("<h3>Play Individual Scene</h3>")

            scene_dropdown = gr.Dropdown(
                choices=[],
                label="Select Scene",
            )
            play_scene_btn = gr.Button("Play Scene", variant="secondary")
            scene_audio = gr.Audio(label="Scene Audio", type="filepath", autoplay=True)
            scene_status = gr.Textbox(label="Scene Status", interactive=False)

            gr.HTML("<hr>")
            gr.HTML("<h3>Full Story</h3>")

            play_all_btn = gr.Button("Play All Scenes", variant="primary", size="lg")
            full_audio = gr.Audio(label="Full Story Audio", type="filepath", autoplay=True, elem_id="full-story-audio")
            full_status = gr.Textbox(
                label="Generation Log",
                interactive=False,
                lines=6,
            )

    # ── Lyrics / Transcript Panel ──────────────────────────────────
    gr.HTML("<hr>")
    gr.HTML("""<h3 style="text-align:center;">Lyrics / Transcript</h3>
        <p style="text-align:center; color:#666; font-size:13px;">
        Who said what, with which mood — follow along as the story plays
        </p>""")
    lyrics_panel = gr.HTML(value='<p style="text-align:center; color:#999; padding:20px;"><em>Generate a story to see the transcript here...</em></p>')

    # ── AI Prompt ───────────────────────────────────────────────────

    AI_PROMPT = """You are a story script writer for a Multi-Voice Story Narrator app. Your job is to write stories in a specific JSON format that the app can read and narrate with different voices.

## JSON FORMAT (follow this EXACTLY):

```json
{
  "title": "Story Title Here",
  "language": "hindi",
  "scenes": [
    {
      "speaker": "character_name",
      "voice": "man",
      "age": "adult",
      "mood": "calm",
      "text": "The dialogue or narration text in the story's language..."
    }
  ]
}
```

## AVAILABLE LANGUAGES (pick ONE per story):
- Indian: "hindi", "kannada", "tamil", "telugu", "bengali", "marathi"
- English accents: "english_us", "english_british", "english_indian", "english_australian", "english_irish"

## AVAILABLE VOICES:
- "man" — male voice
- "woman" — female voice
- "kid" — child voice
- "grandma" — elderly woman voice
- "grandpa" — elderly man voice

## AVAILABLE AGES:
- "child" — for kids
- "young" — young adult
- "adult" — middle-aged
- "elderly" — old person

## AVAILABLE MOODS (changes how they sound):
- "calm" — normal, relaxed
- "angry" — shouting, frustrated
- "happy" — cheerful, laughing
- "sad" — crying, disappointed
- "excited" — energetic, thrilled
- "scared" — fearful, trembling
- "whispering" — quiet, secretive

## RULES:
1. Write ALL dialogue text in the language specified in "language" field
2. Each scene is one character speaking — like a script
3. Use a "narrator" character to describe actions/settings between dialogues
4. Mix different voices, ages, and moods to make it interesting
5. Keep each scene's text to 1-3 sentences max
6. Output ONLY the raw JSON, no explanation or markdown
7. Make stories engaging, emotional, and fun — this is for YouTube content

Now write me a story. I'll tell you the topic, language, and how many characters I want."""

    def _show_prompt():
        return (
            gr.Textbox(value=AI_PROMPT, visible=True),
            gr.Button(visible=True),
        )

    # ── Event handlers ─────────────────────────────────────────────

    # Copy AI prompt
    copy_prompt_btn.click(
        fn=_show_prompt,
        inputs=[],
        outputs=[prompt_output, copy_to_clipboard_btn],
    )

    copy_to_clipboard_btn.click(
        fn=None,
        inputs=[prompt_output],
        js="(text) => { navigator.clipboard.writeText(text); }",
    )

    # Load example
    example_dropdown.change(
        fn=load_example,
        inputs=[example_dropdown],
        outputs=[json_input],
    )

    # Validate JSON
    validate_btn.click(
        fn=validate_json,
        inputs=[json_input],
        outputs=[validation_status, scene_preview],
    ).then(
        fn=update_scene_dropdown,
        inputs=[json_input],
        outputs=[scene_dropdown],
    )

    # Auto-validate and update dropdown when JSON changes
    json_input.change(
        fn=validate_json,
        inputs=[json_input],
        outputs=[validation_status, scene_preview],
    ).then(
        fn=update_scene_dropdown,
        inputs=[json_input],
        outputs=[scene_dropdown],
    )

    # Play single scene
    def _play_scene(json_text, scene_choice, engine):
        if not scene_choice:
            return None, "⚠️ Select a scene first."
        idx = int(scene_choice.split(":")[0])
        return generate_scene_audio(json_text, idx, engine=engine)

    play_scene_btn.click(
        fn=_play_scene,
        inputs=[json_input, scene_dropdown, engine_radio],
        outputs=[scene_audio, scene_status],
    )

    # Play all scenes (JS highlighter is embedded in the lyrics HTML)
    play_all_btn.click(
        fn=generate_all_audio,
        inputs=[json_input, engine_radio],
        outputs=[full_audio, full_status, lyrics_panel],
    )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Multi-Voice Story Narrator")
    print("=" * 50)
    print(f"  Engines: {engine_text}")
    print("\n  Starting server...\n")

    app.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7861,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )
