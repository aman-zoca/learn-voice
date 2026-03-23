"""
TTS API Application
===================
FastAPI-based REST API for text-to-speech synthesis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import io
import uuid
from pathlib import Path
import numpy as np

# API Models
class SynthesizeRequest(BaseModel):
    """Request model for text synthesis."""
    text: str = Field(..., description="Text to synthesize", max_length=5000)
    voice_id: str = Field(default="default", description="Voice ID to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed factor")
    pitch: float = Field(default=0.0, ge=-12, le=12, description="Pitch shift in semitones")
    format: str = Field(default="wav", description="Output format (wav, mp3)")


class SynthesizeResponse(BaseModel):
    """Response model for synthesis."""
    success: bool
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None


class VoiceInfo(BaseModel):
    """Voice information model."""
    id: str
    name: str
    language: str
    gender: Optional[str] = None
    description: Optional[str] = None


class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning."""
    name: str = Field(..., description="Name for the cloned voice")
    description: Optional[str] = Field(None, description="Voice description")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool


# Create FastAPI app
app = FastAPI(
    title="TTS Voice API",
    description="Text-to-Speech API with voice cloning support",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use proper dependency injection)
synthesizer = None
voice_adapter = None
available_voices = {}


# ============================================================================
# Startup and Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global synthesizer, voice_adapter, available_voices

    # Initialize synthesizer
    # In production, load from config/environment
    try:
        from ...02_tts_core.inference.synthesizer import create_synthesizer

        synthesizer = create_synthesizer(
            model_type='tacotron2',
            language='en'
        )

        # Add default voices
        available_voices = {
            'default': VoiceInfo(
                id='default',
                name='Default Voice',
                language='en',
                gender='neutral',
                description='Default English voice'
            )
        }

        print("TTS models loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load TTS models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    pass


# ============================================================================
# Health and Info Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=synthesizer is not None
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "TTS Voice API",
        "version": "0.1.0",
        "docs": "/docs"
    }


# ============================================================================
# Voice Management Endpoints
# ============================================================================

@app.get("/api/v1/voices", response_model=List[VoiceInfo])
async def list_voices():
    """List available voices."""
    return list(available_voices.values())


@app.get("/api/v1/voices/{voice_id}", response_model=VoiceInfo)
async def get_voice(voice_id: str):
    """Get voice details."""
    if voice_id not in available_voices:
        raise HTTPException(status_code=404, detail="Voice not found")

    return available_voices[voice_id]


# ============================================================================
# Synthesis Endpoints
# ============================================================================

@app.post("/api/v1/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text.

    Returns audio file directly.
    """
    if synthesizer is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded"
        )

    try:
        # Synthesize audio
        audio = synthesizer.synthesize(
            text=request.text,
            speed=request.speed,
            pitch_shift=request.pitch
        )

        # Convert to bytes
        audio_bytes = io.BytesIO()

        if request.format == 'wav':
            import scipy.io.wavfile as wav
            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(audio_bytes, 22050, audio_int16)
            media_type = "audio/wav"
        else:
            # Default to WAV
            import scipy.io.wavfile as wav
            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(audio_bytes, 22050, audio_int16)
            media_type = "audio/wav"

        audio_bytes.seek(0)

        return StreamingResponse(
            audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.format}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/synthesize/async", response_model=SynthesizeResponse)
async def synthesize_async(
    request: SynthesizeRequest,
    background_tasks: BackgroundTasks
):
    """
    Asynchronously synthesize speech.

    Returns a job ID that can be used to retrieve the audio later.
    """
    job_id = str(uuid.uuid4())

    # Add synthesis to background tasks
    background_tasks.add_task(
        process_synthesis,
        job_id,
        request.text,
        request.voice_id,
        request.speed
    )

    return SynthesizeResponse(
        success=True,
        message="Synthesis job started",
        audio_url=f"/api/v1/jobs/{job_id}/audio"
    )


async def process_synthesis(
    job_id: str,
    text: str,
    voice_id: str,
    speed: float
):
    """Background synthesis task."""
    # Implementation would save to storage
    pass


# ============================================================================
# Voice Cloning Endpoints
# ============================================================================

@app.post("/api/v1/clone")
async def clone_voice(
    name: str,
    audio_files: List[UploadFile] = File(...),
    description: Optional[str] = None
):
    """
    Clone a voice from audio samples.

    Upload 1-10 audio files (WAV format, 5-30 seconds each).
    """
    if len(audio_files) < 1 or len(audio_files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Please provide 1-10 audio files"
        )

    try:
        # Generate voice ID
        voice_id = f"clone_{uuid.uuid4().hex[:8]}"

        # Process audio files (placeholder)
        # In production, this would:
        # 1. Save audio files
        # 2. Extract speaker embeddings
        # 3. Store in database

        # Add to available voices
        available_voices[voice_id] = VoiceInfo(
            id=voice_id,
            name=name,
            language='en',
            description=description or f"Cloned voice: {name}"
        )

        return {
            "success": True,
            "voice_id": voice_id,
            "message": f"Voice '{name}' cloned successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cloned voice."""
    if voice_id == 'default':
        raise HTTPException(
            status_code=400,
            detail="Cannot delete default voice"
        )

    if voice_id not in available_voices:
        raise HTTPException(status_code=404, detail="Voice not found")

    del available_voices[voice_id]

    return {"success": True, "message": "Voice deleted"}


# ============================================================================
# Streaming Endpoint
# ============================================================================

@app.post("/api/v1/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """
    Stream synthesized audio in chunks.

    Useful for real-time applications.
    """
    if synthesizer is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded"
        )

    async def audio_generator():
        """Generate audio chunks."""
        # This would use a streaming synthesizer
        # For now, generate full audio and chunk it
        audio = synthesizer.synthesize(
            text=request.text,
            speed=request.speed
        )

        # Chunk the audio
        chunk_size = 4096
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]

    return StreamingResponse(
        audio_generator(),
        media_type="audio/raw"
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "05_api.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
