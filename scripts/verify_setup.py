#!/usr/bin/env python3
"""
Setup Verification Script
=========================
Verify that the environment is properly configured for TTS development.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Python Version Check:")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  ❌ Python 3.9+ required")
        return False

    print("  ✅ Python version OK")
    return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\nPyTorch Check:")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠️  CUDA not available (CPU only)")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon) available")

        return True

    except ImportError:
        print("  ❌ PyTorch not installed")
        print("  Run: pip install torch torchaudio")
        return False


def check_audio_libraries():
    """Check audio processing libraries."""
    print("\nAudio Libraries Check:")

    libraries = {
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'scipy': 'scipy',
        'torchaudio': 'torchaudio'
    }

    all_ok = True
    for name, module in libraries.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} not installed")
            all_ok = False

    return all_ok


def check_nlp_libraries():
    """Check NLP libraries."""
    print("\nNLP Libraries Check:")

    libraries = {
        'num2words': 'num2words',
        'unidecode': 'unidecode',
    }

    all_ok = True
    for name, module in libraries.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} not installed (optional)")

    return all_ok


def check_api_libraries():
    """Check API libraries."""
    print("\nAPI Libraries Check:")

    try:
        import fastapi
        print(f"  ✅ FastAPI {fastapi.__version__}")
    except ImportError:
        print("  ⚠️  FastAPI not installed (needed for API)")

    try:
        import uvicorn
        print("  ✅ uvicorn")
    except ImportError:
        print("  ⚠️  uvicorn not installed (needed for API)")

    return True


def check_project_structure():
    """Check project structure."""
    print("\nProject Structure Check:")

    required_dirs = [
        '01_fundamentals',
        '02_tts_core',
        '03_training',
        '04_voice_cloning',
        '05_api',
        'notebooks',
        'tests',
        'scripts'
    ]

    project_root = Path(__file__).parent.parent
    all_ok = True

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ not found")
            all_ok = False

    return all_ok


def run_simple_test():
    """Run a simple test to verify everything works."""
    print("\nSimple Test:")

    try:
        import numpy as np

        # Generate test audio
        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        print(f"  ✅ Generated test audio: {len(audio)} samples")

        # Try to import our modules
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from tts_core.preprocessing.audio.feature_extractor import MelSpectrogramExtractor

            extractor = MelSpectrogramExtractor()
            mel = extractor.mel_spectrogram(audio.astype(np.float32))
            print(f"  ✅ Computed mel spectrogram: {mel.shape}")

        except Exception as e:
            print(f"  ⚠️  Could not test local modules: {e}")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("TTS Environment Verification")
    print("=" * 60)

    results = []

    results.append(("Python Version", check_python_version()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("Audio Libraries", check_audio_libraries()))
    results.append(("NLP Libraries", check_nlp_libraries()))
    results.append(("API Libraries", check_api_libraries()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Simple Test", run_simple_test()))

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All checks passed! Ready for TTS development.")
    else:
        print("⚠️  Some checks failed. Please install missing dependencies.")
        print("\nTo install all dependencies:")
        print("  pip install -r requirements.txt")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
