"""
Voice Adapter Module
====================
Few-shot voice adaptation for voice cloning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from ..encoder.speaker_encoder import SpeakerEncoder, preprocess_wav_for_encoder


class VoiceAdapter:
    """
    Voice adapter for few-shot voice cloning.

    This class handles:
    1. Extracting speaker embeddings from reference audio
    2. Conditioning TTS models with speaker embeddings
    3. Fine-tuning on target speaker (optional)
    """

    def __init__(
        self,
        speaker_encoder: SpeakerEncoder,
        tts_model: nn.Module,
        embedding_dim: int = 256,
        device: str = 'cpu'
    ):
        self.speaker_encoder = speaker_encoder
        self.tts_model = tts_model
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

        self.speaker_encoder.to(self.device)
        self.tts_model.to(self.device)

        # Stored speaker embeddings
        self.speaker_embeddings: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        speaker_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            speaker_id: Optional ID to store the embedding

        Returns:
            Speaker embedding
        """
        self.speaker_encoder.eval()

        # Preprocess audio
        mel = preprocess_wav_for_encoder(audio, sample_rate)
        mel_tensor = torch.FloatTensor(mel).to(self.device)

        # Get embedding
        embedding = self.speaker_encoder.embed_utterance(mel_tensor)

        # Store if ID provided
        if speaker_id is not None:
            self.speaker_embeddings[speaker_id] = embedding

        return embedding

    @torch.no_grad()
    def encode_speaker_from_files(
        self,
        audio_paths: List[str],
        speaker_id: str,
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """
        Extract speaker embedding from multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            speaker_id: ID to store the embedding
            sample_rate: Target sample rate

        Returns:
            Averaged speaker embedding
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required for loading audio")

        embeddings = []

        for path in audio_paths:
            audio, sr = librosa.load(path, sr=sample_rate)
            embedding = self.encode_speaker(audio, sample_rate)
            embeddings.append(embedding)

        # Average embeddings
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        avg_embedding = F.normalize(avg_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)

        self.speaker_embeddings[speaker_id] = avg_embedding

        return avg_embedding

    def get_speaker_embedding(self, speaker_id: str) -> Optional[torch.Tensor]:
        """Get stored speaker embedding."""
        return self.speaker_embeddings.get(speaker_id)

    def synthesize(
        self,
        text: str,
        speaker_id: str,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize speech with a specific speaker's voice.

        Args:
            text: Text to synthesize
            speaker_id: Speaker ID (must have been encoded first)
            **kwargs: Additional arguments for the TTS model

        Returns:
            Audio waveform
        """
        if speaker_id not in self.speaker_embeddings:
            raise ValueError(f"Unknown speaker: {speaker_id}")

        speaker_embedding = self.speaker_embeddings[speaker_id]

        # This depends on the specific TTS model
        # For VITS or multi-speaker Tacotron2:
        audio = self._synthesize_with_embedding(text, speaker_embedding, **kwargs)

        return audio

    def _synthesize_with_embedding(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        **kwargs
    ) -> np.ndarray:
        """
        Synthesize with speaker embedding.

        This is a placeholder - actual implementation depends on the TTS model.
        """
        # Would need to integrate with the specific TTS model
        raise NotImplementedError(
            "Implement based on your TTS model architecture"
        )

    def save_speaker_embeddings(self, path: str):
        """Save speaker embeddings to file."""
        torch.save(self.speaker_embeddings, path)

    def load_speaker_embeddings(self, path: str):
        """Load speaker embeddings from file."""
        self.speaker_embeddings = torch.load(path, map_location=self.device)


class AdaptiveVoiceCloner:
    """
    Adaptive voice cloner with fine-tuning capability.

    For higher quality cloning, this class can fine-tune
    the TTS model on the target speaker's voice.
    """

    def __init__(
        self,
        base_model: nn.Module,
        speaker_encoder: SpeakerEncoder,
        learning_rate: float = 1e-5,
        device: str = 'cpu'
    ):
        self.base_model = base_model
        self.speaker_encoder = speaker_encoder
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        self.base_model.to(self.device)
        self.speaker_encoder.to(self.device)

        # Store adapted models
        self.adapted_models: Dict[str, nn.Module] = {}

    def adapt_to_speaker(
        self,
        speaker_id: str,
        audio_paths: List[str],
        transcripts: List[str],
        num_steps: int = 100,
        batch_size: int = 4
    ):
        """
        Fine-tune the model on a target speaker.

        Args:
            speaker_id: ID for the adapted model
            audio_paths: Paths to reference audio
            transcripts: Corresponding transcripts
            num_steps: Number of fine-tuning steps
            batch_size: Batch size for fine-tuning
        """
        import copy

        # Create a copy of the base model for adaptation
        adapted_model = copy.deepcopy(self.base_model)
        adapted_model.to(self.device)

        # Freeze most layers, only train specific ones
        # This depends on the model architecture
        for param in adapted_model.parameters():
            param.requires_grad = False

        # Unfreeze speaker-specific layers (if any)
        # This is model-specific
        # For example, unfreeze decoder prenet:
        # for param in adapted_model.decoder.prenet.parameters():
        #     param.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapted_model.parameters()),
            lr=self.learning_rate
        )

        # Fine-tuning loop would go here
        # This requires the full training pipeline

        self.adapted_models[speaker_id] = adapted_model

    def synthesize(
        self,
        text: str,
        speaker_id: str,
        use_adapted: bool = True
    ) -> np.ndarray:
        """
        Synthesize with the adapted model.

        Args:
            text: Text to synthesize
            speaker_id: Speaker ID
            use_adapted: Use fine-tuned model if available

        Returns:
            Audio waveform
        """
        if use_adapted and speaker_id in self.adapted_models:
            model = self.adapted_models[speaker_id]
        else:
            model = self.base_model

        # Synthesize using the model
        # Implementation depends on the specific model
        raise NotImplementedError()


class SpeakerEmbeddingAdapter(nn.Module):
    """
    Adapter network to transform speaker embeddings for TTS conditioning.

    This module can learn to transform speaker embeddings from the
    speaker encoder to a format suitable for the TTS model.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 512,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform speaker embedding.

        Args:
            x: Speaker embedding (batch, input_dim)

        Returns:
            Transformed embedding (batch, output_dim)
        """
        return self.adapter(x)


def create_voice_adapter(
    speaker_encoder_path: Optional[str] = None,
    tts_model: Optional[nn.Module] = None,
    device: str = 'cpu'
) -> VoiceAdapter:
    """
    Create a voice adapter.

    Args:
        speaker_encoder_path: Path to speaker encoder weights
        tts_model: TTS model for synthesis
        device: Compute device

    Returns:
        Voice adapter
    """
    from ..encoder.speaker_encoder import create_speaker_encoder

    speaker_encoder = create_speaker_encoder(
        pretrained_path=speaker_encoder_path
    )

    return VoiceAdapter(
        speaker_encoder=speaker_encoder,
        tts_model=tts_model,
        device=device
    )
