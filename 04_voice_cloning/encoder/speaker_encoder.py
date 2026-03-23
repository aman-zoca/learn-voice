"""
Speaker Encoder Module
======================
Extracts speaker embeddings for voice cloning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder network for extracting speaker embeddings.

    Based on the architecture from "Generalized End-to-End Loss for Speaker Verification"
    (GE2E loss). Uses a 3-layer LSTM followed by a projection layer.

    The encoder takes mel spectrograms and produces fixed-size speaker embeddings
    that capture the unique characteristics of a speaker's voice.
    """

    def __init__(
        self,
        n_mels: int = 40,
        hidden_size: int = 256,
        embedding_size: int = 256,
        num_layers: int = 3
    ):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Linear projection to embedding
        self.linear = nn.Linear(hidden_size, embedding_size)

        # For similarity computation
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

    def forward(
        self,
        mels: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute speaker embedding from mel spectrogram.

        Args:
            mels: Mel spectrogram (batch, frames, n_mels)
            hidden: Initial hidden state

        Returns:
            Speaker embedding (batch, embedding_size)
        """
        # LSTM forward pass
        out, (hidden, _) = self.lstm(mels, hidden)

        # Take the last hidden state from the top layer
        embeds_raw = self.linear(hidden[-1])

        # L2 normalize the embeddings
        embeds = F.normalize(embeds_raw, p=2, dim=1)

        return embeds

    def compute_partial_embeddings(
        self,
        mels: torch.Tensor,
        rate: int = 16,
        min_coverage: float = 0.75
    ) -> torch.Tensor:
        """
        Compute embeddings from partial utterances.

        Slides a window over the mel spectrogram and computes
        an embedding for each window. This allows for streaming
        and handles variable-length inputs.

        Args:
            mels: Mel spectrogram (frames, n_mels)
            rate: Number of frames to slide
            min_coverage: Minimum fraction of the utterance to cover

        Returns:
            Partial embeddings (num_partials, embedding_size)
        """
        # Determine window parameters
        n_frames = mels.shape[0]
        frame_step = max(1, n_frames * (1 - min_coverage))
        frame_step = min(frame_step, rate)

        # Sliding window
        wav_slices = []
        steps = max(1, (n_frames - 160) // frame_step + 1)

        for i in range(steps):
            start = int(i * frame_step)
            end = min(start + 160, n_frames)
            wav_slices.append((start, end))

        # Compute embeddings for each slice
        partial_embeds = []
        for start, end in wav_slices:
            mel_slice = mels[start:end].unsqueeze(0)
            if mel_slice.shape[1] >= 10:  # Minimum frames
                embed = self.forward(mel_slice)
                partial_embeds.append(embed)

        if len(partial_embeds) == 0:
            # Return embedding from full utterance
            return self.forward(mels.unsqueeze(0))

        return torch.cat(partial_embeds, dim=0)

    def embed_utterance(
        self,
        mel: torch.Tensor,
        return_partials: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute speaker embedding from a full utterance.

        Args:
            mel: Mel spectrogram (frames, n_mels)
            return_partials: Also return partial embeddings

        Returns:
            Speaker embedding, optionally with partials
        """
        # Compute partial embeddings
        partial_embeds = self.compute_partial_embeddings(mel)

        # Average partial embeddings
        raw_embed = partial_embeds.mean(dim=0)

        # L2 normalize
        embed = F.normalize(raw_embed.unsqueeze(0), p=2, dim=1).squeeze(0)

        if return_partials:
            return embed, partial_embeds
        return embed

    @staticmethod
    def compute_similarity(
        embeds1: torch.Tensor,
        embeds2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.

        Args:
            embeds1: First embeddings (batch1, embedding_size)
            embeds2: Second embeddings (batch2, embedding_size)

        Returns:
            Similarity matrix (batch1, batch2)
        """
        return torch.mm(embeds1, embeds2.T)


class SpeakerVerifier:
    """
    Speaker verification using the speaker encoder.

    Determines if two audio samples are from the same speaker.
    """

    def __init__(
        self,
        encoder: SpeakerEncoder,
        threshold: float = 0.75
    ):
        self.encoder = encoder
        self.threshold = threshold

    def verify(
        self,
        mel1: torch.Tensor,
        mel2: torch.Tensor
    ) -> Tuple[bool, float]:
        """
        Verify if two mel spectrograms are from the same speaker.

        Args:
            mel1: First mel spectrogram
            mel2: Second mel spectrogram

        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        with torch.no_grad():
            embed1 = self.encoder.embed_utterance(mel1)
            embed2 = self.encoder.embed_utterance(mel2)

            similarity = F.cosine_similarity(
                embed1.unsqueeze(0),
                embed2.unsqueeze(0)
            ).item()

        return similarity >= self.threshold, similarity


class GE2ELoss(nn.Module):
    """
    Generalized End-to-End (GE2E) loss for training speaker encoders.

    This loss encourages the embeddings of the same speaker to be close
    together while pushing embeddings of different speakers apart.
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(
        self,
        embeddings: torch.Tensor,
        speakers_per_batch: int,
        utterances_per_speaker: int
    ) -> torch.Tensor:
        """
        Compute GE2E loss.

        Args:
            embeddings: Speaker embeddings (batch, embedding_size)
                        Shape should be (speakers_per_batch * utterances_per_speaker, embedding_size)
            speakers_per_batch: Number of speakers in the batch
            utterances_per_speaker: Number of utterances per speaker

        Returns:
            GE2E loss value
        """
        # Reshape embeddings
        embeddings = embeddings.view(speakers_per_batch, utterances_per_speaker, -1)

        # Compute centroids for each speaker (excluding the current utterance)
        centroids_incl = embeddings.mean(dim=1)

        # Compute similarity matrix
        sim_matrix = []
        for i in range(speakers_per_batch):
            for j in range(utterances_per_speaker):
                # Centroid excluding current utterance
                centroid_excl = (embeddings[i].sum(dim=0) - embeddings[i, j]) / (utterances_per_speaker - 1)

                # Similarity to own centroid
                sim_own = F.cosine_similarity(
                    embeddings[i, j].unsqueeze(0),
                    centroid_excl.unsqueeze(0)
                )

                # Similarity to other centroids
                sim_others = F.cosine_similarity(
                    embeddings[i, j].unsqueeze(0),
                    centroids_incl
                )

                sim_row = self.w * sim_others + self.b
                sim_row[i] = self.w * sim_own + self.b

                sim_matrix.append(sim_row)

        sim_matrix = torch.stack(sim_matrix)

        # Softmax loss
        targets = torch.arange(speakers_per_batch, device=embeddings.device)
        targets = targets.repeat_interleave(utterances_per_speaker)

        loss = F.cross_entropy(sim_matrix, targets)

        return loss


def preprocess_wav_for_encoder(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 40
) -> np.ndarray:
    """
    Preprocess audio for the speaker encoder.

    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        n_mels: Number of mel bands

    Returns:
        Mel spectrogram
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for preprocessing")

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,
        n_mels=n_mels
    )

    # Convert to log scale
    mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    return mel.T  # Return (frames, n_mels)


def create_speaker_encoder(
    n_mels: int = 40,
    hidden_size: int = 256,
    embedding_size: int = 256,
    pretrained_path: Optional[str] = None
) -> SpeakerEncoder:
    """
    Create a speaker encoder.

    Args:
        n_mels: Number of mel bands
        hidden_size: LSTM hidden size
        embedding_size: Output embedding size
        pretrained_path: Path to pretrained weights

    Returns:
        Speaker encoder model
    """
    encoder = SpeakerEncoder(
        n_mels=n_mels,
        hidden_size=hidden_size,
        embedding_size=embedding_size
    )

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        encoder.load_state_dict(checkpoint)

    return encoder
