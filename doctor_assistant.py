"""
Doctor AI Assistant - Continuous Voice Chat
============================================
Just speak naturally - AI responds automatically!
"""

import gradio as gr
import asyncio
import tempfile
import os
import speech_recognition as sr
from datetime import datetime
import requests
import json
import time
import threading
import pygame
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Initialize pygame for audio playback
try:
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False

# ============================================
# Configuration
# ============================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
conversation_active = False  # Flag to control continuous conversation

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
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
}

# ============================================
# Patient Data
# ============================================

patient_data = {
    "name": None, "age": None, "weight": None, "symptoms": [],
    "duration_sick": None, "appointment_date": None, "appointment_time": None,
    "language": "en", "conversation_history": []
}

def reset_patient():
    global patient_data
    patient_data = {
        "name": None, "age": None, "weight": None, "symptoms": [],
        "duration_sick": None, "appointment_date": None, "appointment_time": None,
        "language": "en", "conversation_history": []
    }

def get_summary():
    lines = []
    if patient_data["name"]: lines.append(f"👤 Name: {patient_data['name']}")
    if patient_data["age"]: lines.append(f"📅 Age: {patient_data['age']}")
    if patient_data["weight"]: lines.append(f"⚖️ Weight: {patient_data['weight']}")
    if patient_data["symptoms"]: lines.append(f"🤒 Symptoms: {', '.join(patient_data['symptoms'])}")
    if patient_data["duration_sick"]: lines.append(f"⏱️ Duration: {patient_data['duration_sick']}")
    if patient_data["appointment_date"]: lines.append(f"📆 Appointment: {patient_data['appointment_date']} {patient_data['appointment_time'] or ''}")
    return "\n".join(lines) if lines else "Waiting for patient info..."

# ============================================
# Speech Recognition (Google - more reliable)
# ============================================

# Valid languages we expect
VALID_LANGS = {"en", "hi", "kn", "ta", "te", "ml", "bn", "mr", "gu"}

def is_valid_text(text):
    """Check if text looks valid (not hallucinated)."""
    if not text or len(text) < 2:
        return False
    # Reject if too many repeated characters
    if any(c * 4 in text for c in text):
        return False
    # Reject very short gibberish
    if len(text) < 3 and not text.isalpha():
        return False
    return True

def listen_and_transcribe():
    """Listen to microphone and transcribe."""
    recognizer = sr.Recognizer()

    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0  # Wait 1 sec of silence
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.6

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("🎤 Listening...")

            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Processing...")

            text = None
            detected_lang = "en"

            # Try English first (most reliable)
            try:
                text = recognizer.recognize_google(audio, language="en-IN")
                if text and is_valid_text(text):
                    detected_lang = "en"
                    if LANGDETECT_AVAILABLE:
                        try:
                            dl = detect(text)
                            if dl in VALID_LANGS:
                                detected_lang = dl
                        except:
                            pass
                    print(f"Got: {text} [{detected_lang}]")
                    return text, detected_lang
            except sr.UnknownValueError:
                pass

            # Try Hindi
            try:
                text = recognizer.recognize_google(audio, language="hi-IN")
                if text and is_valid_text(text):
                    print(f"Hindi: {text}")
                    return text, "hi"
            except:
                pass

            # Try Kannada
            try:
                text = recognizer.recognize_google(audio, language="kn-IN")
                if text and is_valid_text(text):
                    print(f"Kannada: {text}")
                    return text, "kn"
            except:
                pass

            # Try Telugu
            try:
                text = recognizer.recognize_google(audio, language="te-IN")
                if text and is_valid_text(text):
                    print(f"Telugu: {text}")
                    return text, "te"
            except:
                pass

            # Try Tamil
            try:
                text = recognizer.recognize_google(audio, language="ta-IN")
                if text and is_valid_text(text):
                    print(f"Tamil: {text}")
                    return text, "ta"
            except:
                pass

            return None, "en"

    except sr.WaitTimeoutError:
        print("Timeout")
        return None, "en"
    except sr.UnknownValueError:
        print("Could not understand")
        return None, "en"
    except Exception as e:
        print(f"Error: {e}")
        return None, "en"

# ============================================
# Doctor AI Response
# ============================================

DOCTOR_PROMPT = """You are Dr. AI. Patient data: {patient_info}

Patient said: "{user_message}"

RULES:
1. If name is null and patient said name → save it, ask age
2. If age is null and patient said age (sal/saal/years/varsha) → save it, ask weight
3. If weight is null and patient said weight → save it, ask symptoms
4. If symptoms is null → ask symptoms
5. Then ask duration, then book appointment

CRITICAL:
- Reply in {language} using NATIVE SCRIPT (Hindi=देवनागरी, Kannada=ಕನ್ನಡ, Tamil=தமிழ், Telugu=తెలుగు)
- NEVER use Romanized text like "aapki umr" - use "आपकी उम्र"
- MAX 6 words
- NEVER ask for info already collected

Response:"""


def get_doctor_response(user_text: str, lang_code: str) -> str:
    """Get response from Doctor AI."""
    global patient_data

    if not OPENAI_API_KEY:
        return "Please connect your API key first."

    # Extract info FIRST so AI sees updated data
    extract_info(user_text)

    patient_data["conversation_history"].append(f"Patient: {user_text}")
    patient_data["language"] = lang_code

    # Build patient info as compact JSON
    info = json.dumps({
        "name": patient_data['name'],
        "age": patient_data['age'],
        "weight": patient_data['weight'],
        "symptoms": patient_data['symptoms'],
        "duration": patient_data['duration_sick'],
        "appointment": patient_data['appointment_date']
    })

    lang_name = LANGUAGE_NAMES.get(lang_code, "English")

    prompt = DOCTOR_PROMPT.format(
        patient_info=info,
        user_message=user_text,
        language=lang_name
    )

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json={"model": "gpt-4.1-mini", "input": prompt},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            bot_text = result.get("output", [{}])[0].get("content", [{}])[0].get("text", "")
            if not bot_text:
                bot_text = "I'm sorry, could you repeat that?"

            patient_data["conversation_history"].append(f"Doctor: {bot_text}")
            return bot_text
        else:
            return "Sorry, I'm having trouble responding. Please try again."

    except Exception as e:
        print(f"API Error: {e}")
        return "Sorry, there was an error. Please try again."


def extract_info(text: str):
    """Extract patient info from text."""
    global patient_data
    import re
    text_lower = text.lower()

    # Name - multiple patterns for Hindi/English
    if not patient_data["name"]:
        name_patterns = [
            r"(?:my name is|i am|i'm|this is|naam|mera naam|mera naam hai)\s+(\w+)",
            r"(\w+)\s+(?:hai mera naam|is my name)",
            r"naam\s+(?:hai\s+)?(\w+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).strip(".,!?")
                if len(name) > 1 and name not in ["is", "hai", "mera", "my", "the"]:
                    patient_data["name"] = name.title()
                    print(f"Extracted name: {patient_data['name']}")
                    break

    # Age - multiple patterns
    if not patient_data["age"]:
        age_patterns = [
            r'(\d{1,3})\s*(?:years?|yrs?|sal|saal|साल|ವರ್ಷ|வயது|varsha)',
            r'(?:age|umra|umar|meri umra|meri umar)\s*(?:is|hai|)?\s*(\d{1,3})',
            r'(\d{1,3})\s*(?:years?|sal|saal)\s*(?:old|hai|ka|ki)?',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = match.group(1)
                if 1 <= int(age) <= 120:
                    patient_data["age"] = age + " years"
                    print(f"Extracted age: {patient_data['age']}")
                    break

    # Weight
    if not patient_data["weight"]:
        match = re.search(r'(\d{2,3})\s*(?:kg|kgs?|kilo|किलो|ಕೆಜಿ)', text_lower)
        if match:
            patient_data["weight"] = match.group(1) + " kg"
            print(f"Extracted weight: {patient_data['weight']}")

    # Duration
    if not patient_data["duration_sick"]:
        duration_patterns = [
            r'(\d+)\s*(?:days?|din|दिन|ದಿನ|நாள்)',
            r'(\d+)\s*(?:weeks?|hafta|hafte)',
            r'(\d+)\s*(?:months?|mahine|mahina)',
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, text_lower)
            if match:
                patient_data["duration_sick"] = match.group(0)
                print(f"Extracted duration: {patient_data['duration_sick']}")
                break

    # Symptoms
    symptoms = ["fever", "cold", "cough", "headache", "pain", "tired", "vomiting", "nausea",
                "bukhar", "bukhaar", "sir dard", "sardi", "khansi", "dard",
                "दर्द", "बुखार", "सर्दी", "खांसी", "सिरदर्द",
                "ನೋವು", "ಜ್ವರ", "தலைவலி", "காய்ச்சல்"]
    for s in symptoms:
        if s in text_lower and s not in patient_data["symptoms"]:
            patient_data["symptoms"].append(s)
            print(f"Extracted symptom: {s}")

    # Appointment
    if "today" in text_lower or "aaj" in text_lower or "आज" in text_lower:
        patient_data["appointment_date"] = "Today"
    elif "tomorrow" in text_lower or "kal" in text_lower or "कल" in text_lower:
        patient_data["appointment_date"] = "Tomorrow"

    if "morning" in text_lower or "subah" in text_lower or "सुबह" in text_lower:
        patient_data["appointment_time"] = "Morning (9-12)"
    elif "afternoon" in text_lower or "dopahar" in text_lower or "evening" in text_lower or "shaam" in text_lower:
        patient_data["appointment_time"] = "Afternoon (2-5)"

# ============================================
# Text-to-Speech
# ============================================

async def tts_async(text: str, lang_code: str) -> str:
    voice = VOICE_MAP.get(lang_code, "en-US-AriaNeural")
    output = os.path.join(tempfile.gettempdir(), "doctor_voice.mp3")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output)
    return output

def speak(text: str, lang_code: str) -> str:
    if EDGE_TTS_AVAILABLE:
        return asyncio.run(tts_async(text, lang_code))
    return None

def play_audio_and_wait(audio_path: str):
    """Play audio and wait for it to finish."""
    if audio_path and PYGAME_AVAILABLE and os.path.exists(audio_path):
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Audio playback error: {e}")

# ============================================
# Main Conversation Function
# ============================================

def single_turn_conversation():
    """Single turn: listen -> respond -> return results."""
    if not OPENAI_API_KEY:
        return None, None, None, "❌ Please connect API key first"

    # Listen to user
    user_text, lang_code = listen_and_transcribe()

    if not user_text:
        return None, None, None, "🎤 Didn't hear anything..."

    # Get doctor response
    doctor_text = get_doctor_response(user_text, lang_code)

    # Convert to speech
    audio_path = speak(doctor_text, lang_code)

    return user_text, doctor_text, audio_path, lang_code


def continuous_conversation(chat_history, status_box):
    """Continuous conversation loop - keeps listening and responding."""
    global conversation_active
    chat_history = chat_history or []

    if not OPENAI_API_KEY:
        yield chat_history, None, get_summary(), "❌ Please connect API key first"
        return

    conversation_active = True
    yield chat_history, None, get_summary(), "🎤 Listening... (speak now)"

    while conversation_active:
        # Listen to user
        user_text, lang_code = listen_and_transcribe()

        if not user_text:
            if conversation_active:
                yield chat_history, None, get_summary(), "🎤 Listening... (speak now)"
            continue

        # Update chat with user message
        chat_history.append({"role": "user", "content": f"🎤 {user_text}"})
        yield chat_history, None, get_summary(), f"🤔 Processing ({LANGUAGE_NAMES.get(lang_code, 'Unknown')})..."

        # Get doctor response
        doctor_text = get_doctor_response(user_text, lang_code)

        # Update chat with doctor response
        chat_history.append({"role": "assistant", "content": f"👨‍⚕️ {doctor_text}"})

        # Convert to speech
        audio_path = speak(doctor_text, lang_code)

        yield chat_history, audio_path, get_summary(), f"🔊 Speaking..."

        # Play audio and wait for it to finish
        if audio_path:
            play_audio_and_wait(audio_path)

        # Small pause before listening again
        time.sleep(0.5)

        if conversation_active:
            yield chat_history, None, get_summary(), "🎤 Listening... (speak now)"

    yield chat_history, None, get_summary(), "⏹️ Conversation stopped"


def stop_conversation():
    """Stop the continuous conversation."""
    global conversation_active
    conversation_active = False
    return "⏹️ Stopping..."


def start_conversation(chat_history):
    """Start listening and have conversation (single turn for compatibility)."""
    chat_history = chat_history or []

    if not OPENAI_API_KEY:
        return chat_history, None, get_summary(), "❌ Please connect API key first"

    # Listen to user
    user_text, lang_code = listen_and_transcribe()

    if not user_text:
        return chat_history, None, get_summary(), "🎤 Didn't hear anything. Click again to speak."

    # Get doctor response
    doctor_text = get_doctor_response(user_text, lang_code)

    # Convert to speech
    audio_path = speak(doctor_text, lang_code)

    # Update chat
    chat_history.append({"role": "user", "content": f"🎤 {user_text}"})
    chat_history.append({"role": "assistant", "content": f"👨‍⚕️ {doctor_text}"})

    return chat_history, audio_path, get_summary(), f"✅ {LANGUAGE_NAMES.get(lang_code, 'Unknown')} detected"


def process_text(text, chat_history):
    """Process typed text."""
    chat_history = chat_history or []

    if not text or not text.strip():
        return chat_history, None, get_summary(), "Type something!", ""

    if not OPENAI_API_KEY:
        return chat_history, None, get_summary(), "❌ Connect API key first", ""

    # Detect language
    lang_code = "en"
    if LANGDETECT_AVAILABLE:
        try:
            lang_code = detect(text)
        except:
            pass

    # Get response
    doctor_text = get_doctor_response(text, lang_code)
    audio_path = speak(doctor_text, lang_code)

    chat_history.append({"role": "user", "content": text})
    chat_history.append({"role": "assistant", "content": f"👨‍⚕️ {doctor_text}"})

    return chat_history, audio_path, get_summary(), "✅ Sent", ""


def connect_api(key):
    global OPENAI_API_KEY
    if key and key.strip():
        OPENAI_API_KEY = key.strip()
    if OPENAI_API_KEY:
        return "✅ Connected! Click Start to talk"
    return "❌ Enter API key or add to .env file"


def get_initial_status():
    if OPENAI_API_KEY:
        return "✅ API key loaded from .env"
    return "Enter API key or add OPENAI_API_KEY to .env"


def new_session():
    reset_patient()
    return [], None, get_summary(), "🔄 New session started"

# ============================================
# UI
# ============================================

def create_ui():
    with gr.Blocks(title="Dr. AI") as app:

        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e88e5 0%, #00acc1 100%); border-radius: 15px;">
            <h1 style="color: white; margin: 0;">👨‍⚕️ Dr. AI</h1>
            <p style="color: #e0f7fa; margin: 5px 0;">Click the microphone and speak naturally - I'll listen and respond!</p>
        </div>
        """)

        # Status Row
        with gr.Row():
            status = gr.Textbox(label="Status", value=get_initial_status(), interactive=False)

        with gr.Row():
            # Chat Column
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                audio_out = gr.Audio(label="Doctor's Response", autoplay=False, type="filepath")

                # Conversation Control Buttons
                with gr.Row():
                    start_btn = gr.Button("🎤 Start Conversation", variant="primary", size="lg", scale=2)
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg", scale=1)

                gr.HTML("<p style='text-align:center; color:#666;'>Click Start → Speak naturally → AI responds → Keep talking!</p>")

                # Text input as backup
                with gr.Row():
                    text_in = gr.Textbox(label="Or type here", placeholder="Type message...", scale=4)
                    send_btn = gr.Button("Send", scale=1)

                new_btn = gr.Button("🔄 New Consultation")

            # Patient Info Column
            with gr.Column(scale=1):
                gr.HTML("<h3>📋 Patient Info</h3>")
                patient_info = gr.Textbox(label="Collected Data", value=get_summary(), interactive=False, lines=10)

        # Events
        start_btn.click(
            fn=continuous_conversation,
            inputs=[chatbot, status],
            outputs=[chatbot, audio_out, patient_info, status]
        )

        stop_btn.click(
            fn=stop_conversation,
            outputs=[status]
        )

        send_btn.click(
            fn=process_text,
            inputs=[text_in, chatbot],
            outputs=[chatbot, audio_out, patient_info, status, text_in]
        )

        text_in.submit(
            fn=process_text,
            inputs=[text_in, chatbot],
            outputs=[chatbot, audio_out, patient_info, status, text_in]
        )

        new_btn.click(fn=new_session, outputs=[chatbot, audio_out, patient_info, status])

        gr.HTML("""
        <div style="text-align: center; margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 8px;">
            <b>Dr. AI collects:</b> Name → Age → Weight → Symptoms → Duration → Appointment
        </div>
        """)

    return app


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  👨‍⚕️ Dr. AI - Medical Assistant")
    print("="*50)
    print("\n→ http://localhost:7863")
    print("\nClick 🎤 and speak naturally!")
    print("="*50 + "\n")

    app = create_ui()
    app.launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=7863)
