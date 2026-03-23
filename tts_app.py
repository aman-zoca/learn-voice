"""
TTS Web Application
===================
Simple UI to enter text and hear it spoken.
"""

import gradio as gr
import tempfile
import os
from pathlib import Path

# Try different TTS backends
TTS_ENGINE = None

# Try gTTS (Google Text-to-Speech) - works offline with internet
try:
    from gtts import gTTS
    TTS_ENGINE = "gtts"
    print("✅ Using Google TTS (gTTS)")
except ImportError:
    pass

# Try pyttsx3 (offline, uses system voices)
if TTS_ENGINE is None:
    try:
        import pyttsx3
        TTS_ENGINE = "pyttsx3"
        print("✅ Using pyttsx3 (system voices)")
    except ImportError:
        pass

# Fallback to macOS 'say' command
if TTS_ENGINE is None:
    import platform
    if platform.system() == "Darwin":
        TTS_ENGINE = "macos_say"
        print("✅ Using macOS 'say' command")

if TTS_ENGINE is None:
    print("❌ No TTS engine available!")
    print("   Install: pip install gtts")


def text_to_speech(text: str, language: str = "en", speed: float = 1.0) -> str:
    """
    Convert text to speech and return audio file path.

    Args:
        text: Text to speak
        language: Language code (en, hi, etc.)
        speed: Speech speed (0.5 to 2.0)

    Returns:
        Path to generated audio file
    """
    if not text.strip():
        return None

    # Create temp file for audio
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "tts_output.mp3")

    try:
        if TTS_ENGINE == "gtts":
            # Google TTS
            tts = gTTS(text=text, lang=language, slow=(speed < 0.8))
            tts.save(output_path)

        elif TTS_ENGINE == "pyttsx3":
            # System TTS
            engine = pyttsx3.init()
            engine.setProperty('rate', int(150 * speed))

            # Save to file
            output_path = output_path.replace('.mp3', '.wav')
            engine.save_to_file(text, output_path)
            engine.runAndWait()

        elif TTS_ENGINE == "macos_say":
            # macOS say command
            output_path = output_path.replace('.mp3', '.aiff')
            rate = int(180 * speed)
            os.system(f'say -r {rate} -o "{output_path}" "{text}"')

        else:
            return None

        return output_path

    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def speak(text: str, language: str, speed: float):
    """Gradio interface function."""
    if not text:
        return None, "⚠️ Please enter some text!"

    audio_path = text_to_speech(text, language, speed)

    if audio_path and os.path.exists(audio_path):
        return audio_path, f"✅ Generated speech for: \"{text[:50]}{'...' if len(text) > 50 else ''}\""
    else:
        return None, "❌ Failed to generate speech. Please try again."


# Language options
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-CN",
    "Arabic": "ar",
}

# Create Gradio Interface
with gr.Blocks(
    title="🎙️ Text-to-Speech",
    theme=gr.themes.Soft(),
    css="""
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        .output-box {
            min-height: 100px;
        }
    """
) as app:

    gr.HTML("""
        <div class="main-title">
            <h1>🎙️ Text-to-Speech</h1>
            <p>Enter text below and click 'Speak' to hear it!</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Text input
            text_input = gr.Textbox(
                label="📝 Enter Text",
                placeholder="Type or paste your text here...",
                lines=5,
                max_lines=10,
            )

            with gr.Row():
                # Language selector
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="🌍 Language",
                    scale=1
                )

                # Speed slider
                speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="⚡ Speed",
                    scale=1
                )

            # Speak button
            speak_btn = gr.Button(
                "🔊 Speak",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            # Audio output
            audio_output = gr.Audio(
                label="🎵 Audio Output",
                type="filepath",
                autoplay=True,
            )

            # Status
            status = gr.Textbox(
                label="Status",
                interactive=False,
            )

    # Example texts
    gr.Examples(
        examples=[
            ["Hello! Welcome to the text-to-speech application.", "English", 1.0],
            ["नमस्ते! आप कैसे हैं?", "Hindi", 1.0],
            ["The quick brown fox jumps over the lazy dog.", "English", 1.2],
            ["Artificial intelligence is transforming the world.", "English", 0.9],
            ["Bonjour! Comment allez-vous?", "French", 1.0],
        ],
        inputs=[text_input, language, speed],
        label="📚 Example Texts"
    )

    # Connect button to function
    speak_btn.click(
        fn=lambda text, lang, spd: speak(text, LANGUAGES.get(lang, "en"), spd),
        inputs=[text_input, language, speed],
        outputs=[audio_output, status]
    )

    # Also trigger on Enter key
    text_input.submit(
        fn=lambda text, lang, spd: speak(text, LANGUAGES.get(lang, "en"), spd),
        inputs=[text_input, language, speed],
        outputs=[audio_output, status]
    )

    gr.HTML("""
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>Built with 💜 using Gradio and gTTS</p>
            <p>Part of the <strong>Learn Voice</strong> TTS project</p>
        </div>
    """)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎙️  Text-to-Speech Application")
    print("="*50)
    print(f"TTS Engine: {TTS_ENGINE}")
    print("\nStarting server...")
    print("Open the URL below in your browser!\n")

    app.launch(
        share=False,  # Set to True to get a public URL
        inbrowser=True,  # Auto-open browser
        server_name="0.0.0.0",
        server_port=7860,
    )
