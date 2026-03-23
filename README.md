# Learn Voice - TTS Voice AI System

A production-grade Text-to-Speech system built from scratch, supporting Hindi + English with multiple accents and voice cloning capabilities.

## Features

- **Multi-language Support**: Hindi and English with proper accent handling
- **Voice Cloning**: Clone any voice with just 30 seconds of audio
- **Production API**: FastAPI-based REST API for easy integration
- **Modern Architecture**: Implements VITS, Tacotron2, and HiFi-GAN
- **Learning-focused**: Comprehensive notebooks and documentation for understanding TTS fundamentals

## Project Structure

```
learn-voice/
├── 01_fundamentals/           # Learning modules
│   ├── 01_python_numpy/       # NumPy and Python basics
│   ├── 02_pytorch_basics/     # PyTorch fundamentals
│   ├── 03_audio_processing/   # Audio signal processing
│   └── 04_neural_networks/    # Neural network concepts
│
├── 02_tts_core/               # Core TTS system
│   ├── models/
│   │   ├── text_encoder/      # Text to embeddings
│   │   ├── acoustic_model/    # Embeddings to mel spectrograms
│   │   ├── vocoder/           # Mel to audio waveform
│   │   ├── vits/              # End-to-end VITS model
│   │   └── multilingual/      # Multi-language support
│   ├── preprocessing/
│   │   ├── text/              # Text normalization, phonemes
│   │   └── audio/             # Audio processing, features
│   └── inference/             # Synthesis pipeline
│
├── 03_training/               # Training infrastructure
│   ├── datasets/              # English and Hindi datasets
│   ├── configs/               # Training configurations
│   ├── trainers/              # Training loops
│   └── checkpoints/           # Saved models
│
├── 04_voice_cloning/          # Voice cloning module
│   ├── encoder/               # Speaker embedding
│   └── adaptation/            # Few-shot voice adaptation
│
├── 05_api/                    # Production API
│   ├── app/                   # FastAPI application
│   └── docker/                # Containerization
│
├── notebooks/                 # Jupyter notebooks for learning
├── tests/                     # Unit and integration tests
└── scripts/                   # Utility scripts
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- FFmpeg (for audio processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/learn-voice.git
cd learn-voice
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

5. Verify GPU setup:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Synthesize Speech

```python
from tts_core.inference.synthesizer import Synthesizer

# Initialize synthesizer
synth = Synthesizer(model_path="checkpoints/vits_english.pt")

# Generate speech
audio = synth.synthesize("Hello, this is a test of the TTS system.")
audio.save("output.wav")
```

### Voice Cloning

```python
from voice_cloning.encoder import SpeakerEncoder
from tts_core.inference.synthesizer import Synthesizer

# Extract speaker embedding from reference audio
encoder = SpeakerEncoder()
speaker_embedding = encoder.embed_utterance("reference_voice.wav")

# Synthesize with cloned voice
synth = Synthesizer(model_path="checkpoints/vits_multispeaker.pt")
audio = synth.synthesize(
    "This will sound like the reference voice.",
    speaker_embedding=speaker_embedding
)
```

### API Usage

Start the API server:
```bash
uvicorn 05_api.app.main:app --host 0.0.0.0 --port 8000
```

Make a synthesis request:
```bash
curl -X POST "http://localhost:8000/api/v1/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_id": "default"}' \
  --output output.wav
```

## Learning Path

This project is designed as a learning journey. Follow these phases:

### Phase 1-2: Foundations (Week 1-4)
- Work through `01_fundamentals/` modules
- Complete notebooks in `notebooks/01_*.ipynb` to `notebooks/04_*.ipynb`
- Understand audio signals, spectrograms, and basic neural networks

### Phase 3-4: Text & Acoustic Modeling (Week 5-10)
- Implement text preprocessing in `02_tts_core/preprocessing/text/`
- Build Tacotron2 in `02_tts_core/models/acoustic_model/`
- Train on LJSpeech dataset

### Phase 5-6: Vocoder & Modern Architectures (Week 11-16)
- Implement HiFi-GAN vocoder
- Build VITS end-to-end model
- Achieve natural-sounding synthesis

### Phase 7-8: Multi-language & Voice Cloning (Week 17-24)
- Add Hindi support
- Implement speaker encoder
- Build voice cloning pipeline

### Phase 9: Production (Week 25-28)
- Deploy FastAPI service
- Add monitoring and scaling
- Optimize for production

## Datasets

### English
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) - 24 hours, single speaker
- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) - 110 speakers
- [LibriTTS](https://openslr.org/60/) - 585 hours

### Hindi
- [IITM Hindi TTS](https://www.iitm.ac.in/donlab/tts/) - Hindi speech corpus
- [Common Voice Hindi](https://commonvoice.mozilla.org/en/datasets)
- [IndicTTS](https://www.iitm.ac.in/donlab/tts/database.php)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.x |
| Audio Processing | librosa, torchaudio |
| Text Processing | epitran, indic-nlp |
| API | FastAPI |
| Database | PostgreSQL |
| Cache | Redis |
| Deployment | Docker |
| Monitoring | Prometheus, Grafana |

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) - Reference implementation
- [NVIDIA Tacotron2](https://github.com/NVIDIA/tacotron2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [VITS](https://github.com/jaywalnut310/vits)
