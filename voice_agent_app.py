"""
Emotional Voice Agent
=====================
AI Voice Agent with emotions, Indian languages, and voice styles.
Uses Microsoft Edge TTS (neural voices) - works offline-ish.

Features:
- 10+ Indian languages (Hindi, Kannada, Tamil, Telugu, etc.)
- Multiple voice styles (cheerful, sad, angry, excited, etc.)
- Male and female voices
- Natural, human-like prosody
- Adjustable speed and pitch
"""

import gradio as gr
import asyncio
import tempfile
import os
import edge_tts
from pathlib import Path

# ============================================
# Voice Configuration
# ============================================

# Indian Language Voices with emotions
VOICES = {
    # Hindi Voices
    "Hindi - Swara (Female)": {"name": "hi-IN-SwaraNeural", "lang": "Hindi", "gender": "Female"},
    "Hindi - Madhur (Male)": {"name": "hi-IN-MadhurNeural", "lang": "Hindi", "gender": "Male"},

    # Kannada Voices
    "Kannada - Sapna (Female)": {"name": "kn-IN-SapnaNeural", "lang": "Kannada", "gender": "Female"},
    "Kannada - Gagan (Male)": {"name": "kn-IN-GaganNeural", "lang": "Kannada", "gender": "Male"},

    # Tamil Voices
    "Tamil - Pallavi (Female)": {"name": "ta-IN-PallaviNeural", "lang": "Tamil", "gender": "Female"},
    "Tamil - Valluvar (Male)": {"name": "ta-IN-ValluvarNeural", "lang": "Tamil", "gender": "Male"},

    # Telugu Voices
    "Telugu - Shruti (Female)": {"name": "te-IN-ShrutiNeural", "lang": "Telugu", "gender": "Female"},
    "Telugu - Mohan (Male)": {"name": "te-IN-MohanNeural", "lang": "Telugu", "gender": "Male"},

    # Malayalam Voices
    "Malayalam - Sobhana (Female)": {"name": "ml-IN-SobhanaNeural", "lang": "Malayalam", "gender": "Female"},
    "Malayalam - Midhun (Male)": {"name": "ml-IN-MidhunNeural", "lang": "Malayalam", "gender": "Male"},

    # Bengali Voices
    "Bengali - Tanishaa (Female)": {"name": "bn-IN-TanishaaNeural", "lang": "Bengali", "gender": "Female"},
    "Bengali - Bashkar (Male)": {"name": "bn-IN-BashkarNeural", "lang": "Bengali", "gender": "Male"},

    # Marathi Voices
    "Marathi - Aarohi (Female)": {"name": "mr-IN-AarohiNeural", "lang": "Marathi", "gender": "Female"},
    "Marathi - Manohar (Male)": {"name": "mr-IN-ManoharNeural", "lang": "Marathi", "gender": "Male"},

    # Gujarati Voices
    "Gujarati - Dhwani (Female)": {"name": "gu-IN-DhwaniNeural", "lang": "Gujarati", "gender": "Female"},
    "Gujarati - Niranjan (Male)": {"name": "gu-IN-NiranjanNeural", "lang": "Gujarati", "gender": "Male"},

    # English (Indian) Voices
    "English (India) - Neerja (Female)": {"name": "en-IN-NeerjaNeural", "lang": "English", "gender": "Female"},
    "English (India) - Prabhat (Male)": {"name": "en-IN-PrabhatNeural", "lang": "English", "gender": "Male"},

    # English (US) Voices - Best emotion support
    "English (US) - Jenny (Female)": {"name": "en-US-JennyNeural", "lang": "English", "gender": "Female"},
    "English (US) - Guy (Male)": {"name": "en-US-GuyNeural", "lang": "English", "gender": "Male"},
    "English (US) - Aria (Female)": {"name": "en-US-AriaNeural", "lang": "English", "gender": "Female"},
    "English (US) - Davis (Male)": {"name": "en-US-DavisNeural", "lang": "English", "gender": "Male"},

    # More Indian Languages
    "Punjabi - Female": {"name": "pa-IN-Female", "lang": "Punjabi", "gender": "Female"},
    "Odia - Female": {"name": "or-IN-SubhasiniNeural", "lang": "Odia", "gender": "Female"},
    "Odia - Male": {"name": "or-IN-SukantNeural", "lang": "Odia", "gender": "Male"},
    "Assamese - Female": {"name": "as-IN-YashicaNeural", "lang": "Assamese", "gender": "Female"},
    "Assamese - Male": {"name": "as-IN-PriyomNeural", "lang": "Assamese", "gender": "Male"},
}

# Emotion/Style options (SSML styles supported by neural voices)
EMOTIONS = {
    "Normal": {
        "style": None,
        "description": "Natural, conversational tone"
    },
    "Cheerful / Happy": {
        "style": "cheerful",
        "description": "Upbeat, joyful, positive energy"
    },
    "Excited": {
        "style": "excited",
        "description": "High energy, enthusiastic, pumped up!"
    },
    "Friendly": {
        "style": "friendly",
        "description": "Warm, welcoming, approachable"
    },
    "Hopeful": {
        "style": "hopeful",
        "description": "Optimistic, looking forward"
    },
    "Sad": {
        "style": "sad",
        "description": "Melancholic, sorrowful, emotional"
    },
    "Angry": {
        "style": "angry",
        "description": "Frustrated, intense, forceful"
    },
    "Fearful": {
        "style": "fearful",
        "description": "Nervous, scared, anxious"
    },
    "Shouting": {
        "style": "shouting",
        "description": "Loud, projecting, calling out"
    },
    "Whispering": {
        "style": "whispering",
        "description": "Soft, intimate, secretive"
    },
    "Terrified": {
        "style": "terrified",
        "description": "Extreme fear, panic"
    },
    "Unfriendly": {
        "style": "unfriendly",
        "description": "Cold, dismissive"
    },
    "Empathetic": {
        "style": "empathetic",
        "description": "Understanding, caring, compassionate"
    },
    "Calm": {
        "style": "calm",
        "description": "Peaceful, relaxed, soothing"
    },
    "Gentle": {
        "style": "gentle",
        "description": "Soft, tender, kind"
    },
    "Serious": {
        "style": "serious",
        "description": "Formal, professional, grave"
    },
    "Narration - Professional": {
        "style": "narration-professional",
        "description": "Documentary style, authoritative"
    },
    "Newscast": {
        "style": "newscast",
        "description": "News anchor style, clear and formal"
    },
    "Customer Service": {
        "style": "customerservice",
        "description": "Helpful, polite, service-oriented"
    },
    "Poetry Reading": {
        "style": "poetry-reading",
        "description": "Expressive, rhythmic, artistic"
    },
    "Sports Commentary": {
        "style": "sports-commentary",
        "description": "Energetic, fast-paced, exciting"
    },
    "Documentary": {
        "style": "documentary-narration",
        "description": "Informative, educational"
    },
    "Lyrical": {
        "style": "lyrical",
        "description": "Musical, melodic quality"
    },
}

# Intensity levels for emotions
INTENSITY_LEVELS = ["0.5", "0.75", "1.0", "1.25", "1.5", "2.0"]


# ============================================
# TTS Functions
# ============================================

async def synthesize_async(
    text: str,
    voice: str,
    emotion: str,
    intensity: str,
    speed: float,
    pitch: int
) -> str:
    """Async function to synthesize speech with emotions."""

    voice_info = VOICES.get(voice, VOICES["Hindi - Swara (Female)"])
    voice_name = voice_info["name"]

    # Build rate and pitch strings
    rate_percent = int((speed - 1) * 100)
    rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
    pitch_str = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

    # Output file
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "voice_agent_output.mp3")

    # Direct synthesis - no extra content, just the text
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice_name,
        rate=rate_str,
        pitch=pitch_str
    )
    await communicate.save(output_path)

    return output_path


def generate_speech(
    text: str,
    voice: str,
    emotion: str,
    intensity: str,
    speed: float,
    pitch: int
) -> tuple:
    """Main function to generate emotional speech."""

    if not text.strip():
        return None, "Please enter some text!"

    try:
        # Run async function
        output_path = asyncio.run(synthesize_async(
            text, voice, emotion, intensity, speed, pitch
        ))

        if output_path and os.path.exists(output_path):
            voice_info = VOICES.get(voice, {})
            emotion_info = EMOTIONS.get(emotion, {})

            status = f"Generated: {voice_info.get('lang', 'Unknown')} | "
            status += f"Voice: {voice_info.get('gender', 'Unknown')} | "
            status += f"Style: {emotion}"

            return output_path, status
        else:
            return None, "Failed to generate audio"

    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================
# Gradio UI
# ============================================

def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Voice Agent - Emotional Indian TTS",
        theme=gr.themes.Soft()
    ) as app:

        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">Voice Agent</h1>
            <h3 style="color: #f0f0f0; margin: 10px 0;">Emotional Voice AI for Indian Languages</h3>
            <p style="color: #ddd;">Speak with emotions in Hindi, Kannada, Tamil, Telugu, Bengali & more!</p>
        </div>
        """)

        with gr.Row():
            # Left Column - Input
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter Your Text",
                    placeholder="Type what you want to say...\n\nExamples:\n• Hello! How are you today?\n• नमस्ते! आप कैसे हैं?\n• ನಮಸ್ಕಾರ! ಹೇಗಿದ್ದೀರಾ?",
                    lines=6,
                )

                with gr.Row():
                    voice = gr.Dropdown(
                        choices=list(VOICES.keys()),
                        value="Hindi - Swara (Female)",
                        label="Select Voice",
                    )

                    emotion = gr.Dropdown(
                        choices=list(EMOTIONS.keys()),
                        value="Normal",
                        label="Emotion / Style",
                    )

                with gr.Row():
                    intensity = gr.Dropdown(
                        choices=INTENSITY_LEVELS,
                        value="1.0",
                        label="Emotion Intensity",
                    )

                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speaking Speed",
                    )

                    pitch = gr.Slider(
                        minimum=-50,
                        maximum=50,
                        value=0,
                        step=5,
                        label="Pitch (Hz)",
                    )

                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    size="lg"
                )

            # Right Column - Output
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    autoplay=True,
                )

                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

                emotion_desc = gr.Textbox(
                    label="Emotion Description",
                    value=EMOTIONS["Normal"]["description"],
                    interactive=False,
                )

        # Update emotion description when emotion changes
        def update_desc(emotion_name):
            return EMOTIONS.get(emotion_name, {}).get("description", "")

        emotion.change(
            fn=update_desc,
            inputs=[emotion],
            outputs=[emotion_desc]
        )

        # Generate button
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, voice, emotion, intensity, speed, pitch],
            outputs=[audio_output, status]
        )

        # Enter key to generate
        text_input.submit(
            fn=generate_speech,
            inputs=[text_input, voice, emotion, intensity, speed, pitch],
            outputs=[audio_output, status]
        )

        # ============ Examples ============
        gr.HTML("<h3 style='margin-top: 30px;'>Try These Examples</h3>")

        gr.Examples(
            examples=[
                # Hindi examples
                ["नमस्ते! मैं आपका AI वॉइस असिस्टेंट हूं।", "Hindi - Swara (Female)", "Friendly", "1.0", 1.0, 0],
                ["आज मैं बहुत खुश हूं! क्या बात है!", "Hindi - Madhur (Male)", "Cheerful / Happy", "1.5", 1.1, 5],
                ["यह बहुत दुख की बात है...", "Hindi - Swara (Female)", "Sad", "1.25", 0.9, -5],

                # Kannada examples
                ["ನಮಸ್ಕಾರ! ನಿಮಗೆ ಹೇಗಿದೆ?", "Kannada - Sapna (Female)", "Friendly", "1.0", 1.0, 0],
                ["ಇದು ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ!", "Kannada - Gagan (Male)", "Excited", "1.25", 1.1, 5],

                # Tamil examples
                ["வணக்கம்! எப்படி இருக்கீங்க?", "Tamil - Pallavi (Female)", "Friendly", "1.0", 1.0, 0],
                ["இது மிகவும் அற்புதமான செய்தி!", "Tamil - Valluvar (Male)", "Excited", "1.5", 1.1, 5],

                # Telugu examples
                ["నమస్కారం! మీరు ఎలా ఉన్నారు?", "Telugu - Shruti (Female)", "Friendly", "1.0", 1.0, 0],

                # English with emotions
                ["Hey! I'm so excited to talk to you today!", "English (US) - Aria (Female)", "Excited", "1.5", 1.1, 5],
                ["I have some sad news to share with you...", "English (US) - Jenny (Female)", "Sad", "1.25", 0.85, -10],
                ["Let me tell you a secret...", "English (US) - Aria (Female)", "Whispering", "1.0", 0.8, -5],
                ["THIS IS INCREDIBLE NEWS!", "English (US) - Guy (Male)", "Shouting", "1.5", 1.2, 10],
                ["I understand how you're feeling.", "English (US) - Jenny (Female)", "Empathetic", "1.25", 0.9, -5],
                ["Welcome to today's news broadcast.", "English (US) - Davis (Male)", "Newscast", "1.0", 1.0, 0],

                # More Indian languages
                ["নমস্কার! কেমন আছেন?", "Bengali - Tanishaa (Female)", "Friendly", "1.0", 1.0, 0],
                ["નમસ્તે! કેમ છો?", "Gujarati - Dhwani (Female)", "Friendly", "1.0", 1.0, 0],
                ["नमस्कार! कसे आहात?", "Marathi - Aarohi (Female)", "Friendly", "1.0", 1.0, 0],
            ],
            inputs=[text_input, voice, emotion, intensity, speed, pitch],
            label="Click any example to try it!"
        )

        # ============ Voice Reference ============
        with gr.Accordion("Available Voices Reference", open=False):
            gr.HTML("""
            <div style="padding: 15px;">
                <h4>Indian Language Voices</h4>
                <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                    <tr style="background: #f0f0f0;">
                        <th style="padding: 8px; border: 1px solid #ddd;">Language</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Female Voice</th>
                        <th style="padding: 8px; border: 1px solid #ddd;">Male Voice</th>
                    </tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Hindi</td><td style="padding: 8px; border: 1px solid #ddd;">Swara</td><td style="padding: 8px; border: 1px solid #ddd;">Madhur</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Kannada</td><td style="padding: 8px; border: 1px solid #ddd;">Sapna</td><td style="padding: 8px; border: 1px solid #ddd;">Gagan</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Tamil</td><td style="padding: 8px; border: 1px solid #ddd;">Pallavi</td><td style="padding: 8px; border: 1px solid #ddd;">Valluvar</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Telugu</td><td style="padding: 8px; border: 1px solid #ddd;">Shruti</td><td style="padding: 8px; border: 1px solid #ddd;">Mohan</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Malayalam</td><td style="padding: 8px; border: 1px solid #ddd;">Sobhana</td><td style="padding: 8px; border: 1px solid #ddd;">Midhun</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Bengali</td><td style="padding: 8px; border: 1px solid #ddd;">Tanishaa</td><td style="padding: 8px; border: 1px solid #ddd;">Bashkar</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Marathi</td><td style="padding: 8px; border: 1px solid #ddd;">Aarohi</td><td style="padding: 8px; border: 1px solid #ddd;">Manohar</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Gujarati</td><td style="padding: 8px; border: 1px solid #ddd;">Dhwani</td><td style="padding: 8px; border: 1px solid #ddd;">Niranjan</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Odia</td><td style="padding: 8px; border: 1px solid #ddd;">Subhasini</td><td style="padding: 8px; border: 1px solid #ddd;">Sukant</td></tr>
                    <tr><td style="padding: 8px; border: 1px solid #ddd;">Assamese</td><td style="padding: 8px; border: 1px solid #ddd;">Yashica</td><td style="padding: 8px; border: 1px solid #ddd;">Priyom</td></tr>
                </table>

                <h4 style="margin-top: 20px;">Available Emotions/Styles</h4>
                <p><b>Best for English voices:</b> Cheerful, Excited, Sad, Angry, Whispering, Shouting, Empathetic, Calm, Newscast, Sports Commentary</p>
                <p><b>Note:</b> Not all emotions work with all voices. English (US) voices have the best emotion support.</p>
            </div>
            """)

        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; color: #666;">
            <p>Powered by Microsoft Edge Neural TTS</p>
            <p><b>Voice Agent</b> - Emotional Voice AI for Everyone</p>
        </div>
        """)

    return app


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("     Voice Agent - Emotional Indian TTS")
    print("="*60)
    print("\nFeatures:")
    print("  - 10+ Indian Languages")
    print("  - 20+ Emotion Styles")
    print("  - Male & Female Voices")
    print("  - Adjustable Speed & Pitch")
    print("\nStarting server on http://localhost:7861")
    print("="*60 + "\n")

    app = create_ui()
    app.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7861,
    )
