"""
Voice Chat Agent with ChatGPT
=============================
Talk to AI and it talks back!

Features:
- Speech-to-Text: Understands what you say
- ChatGPT: Smart responses
- Text-to-Speech: Speaks back to you
- All Indian languages supported
"""

import gradio as gr
import asyncio
import tempfile
import os
import speech_recognition as sr
from datetime import datetime
import requests
import json

# Try imports
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except:
    EDGE_TTS_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except:
    LANGDETECT_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# ============================================
# OpenAI Configuration
# ============================================

OPENAI_API_KEY = ""

# ============================================
# Voice Configuration
# ============================================

VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "kn": "kn-IN-SapnaNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "or": "or-IN-SubhasiniNeural",
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "bn": "Bengali", "mr": "Marathi",
    "gu": "Gujarati", "or": "Odia",
}

# Conversation memory
conversation_memory = []

# ============================================
# Speech-to-Text
# ============================================

def transcribe_audio(audio_path: str) -> tuple:
    """Transcribe audio to text."""
    if audio_path is None:
        return None, "en"

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

        # Detect language
        lang_code = "en"
        if LANGDETECT_AVAILABLE and text:
            try:
                lang_code = detect(text)
            except:
                pass
        return text, lang_code
    except:
        return None, "en"


# ============================================
# GPT-4.1-mini Response (Responses API)
# ============================================

def generate_response_chatgpt(user_text: str, lang_code: str) -> str:
    """Generate response using GPT-4.1-mini Responses API."""
    global conversation_memory

    if not OPENAI_API_KEY:
        return generate_response_basic(user_text, lang_code)

    lang_name = LANGUAGE_NAMES.get(lang_code, "English")

    # Build prompt with context
    prompt = f"""You are a friendly voice assistant. The user is speaking in {lang_name}.
ALWAYS respond in {lang_name}. Keep responses short (1-2 sentences) since they will be spoken aloud.

User: {user_text}"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json={
                "model": "gpt-4.1-mini",
                "input": prompt
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            # Extract the response text
            bot_response = result.get("output", [{}])[0].get("content", [{}])[0].get("text", "")

            if not bot_response:
                bot_response = str(result)[:200]

            # Save to memory
            conversation_memory.append({"role": "user", "content": user_text})
            conversation_memory.append({"role": "assistant", "content": bot_response})

            if len(conversation_memory) > 10:
                conversation_memory = conversation_memory[-10:]

            return bot_response
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return generate_response_basic(user_text, lang_code)

    except Exception as e:
        print(f"GPT error: {e}")
        return generate_response_basic(user_text, lang_code)


def generate_response_basic(user_text: str, lang_code: str) -> str:
    """Basic responses when ChatGPT is not available."""
    text_lower = user_text.lower()

    if any(w in text_lower for w in ["hello", "hi", "hey", "namaste", "नमस्ते", "ನಮಸ್ಕಾರ"]):
        responses = {
            "en": "Hello! Please add your OpenAI API key above to enable smart responses!",
            "hi": "नमस्ते! स्मार्ट जवाब के लिए ऊपर OpenAI API key डालें!",
            "kn": "ನಮಸ್ಕಾರ! ಸ್ಮಾರ್ಟ್ ಉತ್ತರಗಳಿಗೆ ಮೇಲೆ API key ಸೇರಿಸಿ!",
        }
        return responses.get(lang_code, responses["en"])

    elif any(w in text_lower for w in ["time", "समय", "ಸಮಯ"]):
        now = datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."

    else:
        return f"I heard: '{user_text[:50]}'. Please add your OpenAI API key to get smart responses!"


def generate_response(user_text: str, lang_code: str) -> str:
    """Main response generator."""
    if OPENAI_API_KEY:
        return generate_response_chatgpt(user_text, lang_code)
    else:
        return generate_response_basic(user_text, lang_code)


# ============================================
# Text-to-Speech
# ============================================

async def text_to_speech_async(text: str, lang_code: str) -> str:
    """Convert text to speech using Edge TTS."""
    voice = VOICE_MAP.get(lang_code, "en-US-AriaNeural")

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "voice_chat_output.mp3")

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)

    return output_path


def text_to_speech(text: str, lang_code: str) -> str:
    """Sync wrapper for TTS."""
    if EDGE_TTS_AVAILABLE:
        return asyncio.run(text_to_speech_async(text, lang_code))
    return None


# ============================================
# Main Chat Functions
# ============================================

def process_voice_input(audio, chat_history):
    """Process voice input and generate response."""
    chat_history = chat_history or []

    if audio is None:
        return chat_history, None, "Please record audio first!"

    # Transcribe
    user_text, lang_code = transcribe_audio(audio)

    if not user_text:
        return chat_history, None, "Could not understand. Please try again."

    lang_name = LANGUAGE_NAMES.get(lang_code, "English")

    # Generate response
    bot_response = generate_response(user_text, lang_code)

    # Text to speech
    audio_path = text_to_speech(bot_response, lang_code)

    # Update chat (Gradio 6 format)
    chat_history.append({"role": "user", "content": f"🎤 {user_text}"})
    chat_history.append({"role": "assistant", "content": bot_response})

    status = f"Language: {lang_name} | GPT: {'Connected' if OPENAI_API_KEY else 'Add API key'}"

    return chat_history, audio_path, status


def process_text_input(text, chat_history, lang_code):
    """Process text input."""
    chat_history = chat_history or []

    if not text or not text.strip():
        return chat_history, None, "Please enter text!", ""

    # Detect language
    if lang_code == "auto" and LANGDETECT_AVAILABLE:
        try:
            lang_code = detect(text)
        except:
            lang_code = "en"
    elif lang_code == "auto":
        lang_code = "en"

    # Generate response
    bot_response = generate_response(text, lang_code)

    # Text to speech
    audio_path = text_to_speech(bot_response, lang_code)

    # Update chat (Gradio 6 format)
    chat_history.append({"role": "user", "content": text})
    chat_history.append({"role": "assistant", "content": bot_response})

    status = f"Language: {LANGUAGE_NAMES.get(lang_code, 'Unknown')} | GPT: {'Connected' if OPENAI_API_KEY else 'Add API key'}"

    return chat_history, audio_path, status, ""


def clear_chat():
    """Clear chat."""
    global conversation_memory
    conversation_memory = []
    return [], None, "Chat cleared!"


def connect_openai(api_key):
    """Connect to OpenAI with API key."""
    global OPENAI_API_KEY

    if not api_key or not api_key.strip():
        return "Please enter your API key"

    OPENAI_API_KEY = api_key.strip()

    # Test the connection with Responses API
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json={
                "model": "gpt-4.1-mini",
                "input": "Hi"
            },
            timeout=10
        )
        if response.status_code == 200:
            return "✅ Connected to GPT-4.1-mini!"
        else:
            return f"❌ Error {response.status_code}: {response.text[:50]}"
    except Exception as e:
        return f"❌ Failed: {str(e)[:50]}"


# ============================================
# Gradio UI
# ============================================

def create_ui():
    with gr.Blocks(title="Voice Chat with ChatGPT") as app:

        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">Voice Chat Agent</h1>
            <h3 style="color: #f0f0f0;">Talk to ChatGPT - It Talks Back!</h3>
        </div>
        """)

        # OpenAI API Key Section
        with gr.Accordion("🔑 OpenAI API Key", open=True):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Enter your OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    scale=3
                )
                connect_btn = gr.Button("Connect", variant="primary", scale=1)
                api_status = gr.Textbox(
                    label="Status",
                    value="Not connected",
                    interactive=False,
                    scale=2
                )

            connect_btn.click(fn=connect_openai, inputs=[api_key_input], outputs=[api_status])

        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                gr.HTML("<h3>🎤 Talk to AI</h3>")

                audio_input = gr.Audio(
                    label="Record Your Voice",
                    sources=["microphone", "upload"],
                    type="filepath",
                )
                voice_btn = gr.Button("🎤 Send Voice", variant="primary", size="lg")

                gr.HTML("<hr><h4>Or Type:</h4>")

                text_input = gr.Textbox(
                    label="Type Message",
                    placeholder="Type in any language...",
                    lines=2,
                )
                lang_select = gr.Dropdown(
                    choices=["auto"] + list(LANGUAGE_NAMES.keys()),
                    value="auto",
                    label="Language",
                )
                text_btn = gr.Button("💬 Send Text")
                clear_btn = gr.Button("🗑️ Clear")

            # Chat Column
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                audio_output = gr.Audio(label="🔊 AI Voice", type="filepath", autoplay=True)
                status = gr.Textbox(label="Status", interactive=False)

        # Events
        voice_btn.click(
            fn=process_voice_input,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, audio_output, status]
        )

        text_btn.click(
            fn=process_text_input,
            inputs=[text_input, chatbot, lang_select],
            outputs=[chatbot, audio_output, status, text_input]
        )

        text_input.submit(
            fn=process_text_input,
            inputs=[text_input, chatbot, lang_select],
            outputs=[chatbot, audio_output, status, text_input]
        )

        clear_btn.click(fn=clear_chat, outputs=[chatbot, audio_output, status])

        # Examples
        gr.Examples(
            examples=[
                ["Hello! How are you?", "auto"],
                ["Tell me a joke", "en"],
                ["नमस्ते! आप कैसे हैं?", "hi"],
                ["ನಮಸ್ಕಾರ!", "kn"],
                ["What is AI?", "en"],
            ],
            inputs=[text_input, lang_select],
        )

        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; color: #666; padding: 15px; background: #f9f9f9; border-radius: 8px;">
            <b>Voice Chat with ChatGPT</b><br>
            Speech Recognition → ChatGPT → Voice Output<br>
            <small>Supports: English, Hindi, Kannada, Tamil, Telugu, Bengali, Marathi, Gujarati</small>
        </div>
        """)

    return app


if __name__ == "__main__":
    print("\n" + "="*60)
    print("     Voice Chat Agent with ChatGPT")
    print("="*60)
    print("\nEnter your OpenAI API key in the UI to start chatting!")
    print("\nStarting server on http://localhost:7862")
    print("="*60 + "\n")

    app = create_ui()
    app.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7862,
    )
