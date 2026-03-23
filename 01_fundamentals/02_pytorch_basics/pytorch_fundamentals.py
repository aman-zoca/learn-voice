"""
PyTorch Fundamentals for TTS
============================
This module covers essential PyTorch concepts needed for building TTS models.

Key Concepts:
- Tensors and GPU operations
- Autograd and gradients
- Building neural network modules
- Training loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List


# =============================================================================
# 1. Tensor Basics
# =============================================================================

def tensor_creation_examples():
    """Demonstrate various ways to create tensors."""

    # From Python list
    tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
    print(f"From list: {tensor_from_list}")

    # From NumPy array
    np_array = np.random.randn(3, 4)
    tensor_from_numpy = torch.from_numpy(np_array)
    print(f"From numpy shape: {tensor_from_numpy.shape}")

    # Zero tensor (like empty audio buffer)
    zeros = torch.zeros(22050, dtype=torch.float32)
    print(f"Zeros shape: {zeros.shape}")

    # Random tensor (useful for testing)
    random_tensor = torch.randn(16, 80, 100)  # batch, mel_bins, time
    print(f"Random tensor shape: {random_tensor.shape}")

    # Specific dtype for audio
    audio_tensor = torch.zeros(16000, dtype=torch.float32)
    print(f"Audio dtype: {audio_tensor.dtype}")


def check_device_availability():
    """Check and use available compute devices."""
    print("Device Availability:")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")

    # Check for MPS (Apple Silicon)
    print(f"  MPS available: {torch.backends.mps.is_available()}")

    # Get best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"  Using device: {device}")
    return device


def tensor_operations_for_audio():
    """Common tensor operations used in audio processing."""

    # Simulated audio batch: (batch_size, samples)
    audio = torch.randn(8, 22050)

    # Reshape for processing: add channel dimension
    audio_with_channel = audio.unsqueeze(1)  # (8, 1, 22050)
    print(f"With channel: {audio_with_channel.shape}")

    # Transpose (useful for attention)
    # (batch, time, features) <-> (batch, features, time)
    features = torch.randn(8, 100, 256)  # batch, time, features
    transposed = features.transpose(1, 2)  # batch, features, time
    print(f"Transposed: {transposed.shape}")

    # Permute for more complex reordering
    # Common in spectrogram processing
    spec = torch.randn(8, 80, 100)  # batch, mel, time
    permuted = spec.permute(0, 2, 1)  # batch, time, mel
    print(f"Permuted: {permuted.shape}")

    # Concatenation (combining features)
    text_features = torch.randn(8, 100, 256)
    speaker_embed = torch.randn(8, 1, 256).expand(-1, 100, -1)
    combined = torch.cat([text_features, speaker_embed], dim=-1)
    print(f"Concatenated: {combined.shape}")

    # Masking (for variable length sequences)
    lengths = torch.tensor([80, 100, 60, 90, 70, 85, 95, 75])
    max_len = 100
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    print(f"Mask shape: {mask.shape}")


# =============================================================================
# 2. Autograd Basics
# =============================================================================

def autograd_example():
    """Demonstrate automatic differentiation."""

    # Create tensor with gradient tracking
    x = torch.tensor([2.0, 3.0], requires_grad=True)

    # Forward pass
    y = x ** 2 + 2 * x + 1
    loss = y.sum()

    print(f"x: {x}")
    print(f"y = x² + 2x + 1: {y}")
    print(f"loss = sum(y): {loss}")

    # Backward pass
    loss.backward()

    # dy/dx = 2x + 2
    print(f"Gradient (2x + 2): {x.grad}")


def gradient_flow_example():
    """Show how gradients flow through operations."""

    # Simulated network computation
    input_data = torch.randn(4, 10, requires_grad=True)

    # Linear transformation (what a layer does)
    weight = torch.randn(10, 20, requires_grad=True)
    bias = torch.randn(20, requires_grad=True)

    # Forward: y = xW + b
    output = torch.matmul(input_data, weight) + bias

    # ReLU activation
    activated = F.relu(output)

    # Loss (simplified)
    loss = activated.mean()

    # Backward
    loss.backward()

    print(f"Input grad shape: {input_data.grad.shape}")
    print(f"Weight grad shape: {weight.grad.shape}")
    print(f"Bias grad shape: {bias.grad.shape}")


# =============================================================================
# 3. Building Neural Network Modules
# =============================================================================

class AudioEncoder(nn.Module):
    """
    Simple audio encoder - encodes waveform to features.

    This demonstrates:
    - nn.Module structure
    - Convolutional layers for audio
    - Layer composition
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 256,
        out_channels: int = 512,
        kernel_size: int = 5
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, hidden_channels,
            kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv3 = nn.Conv1d(
            hidden_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm3 = nn.BatchNorm1d(out_channels)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Audio tensor (batch, channels, samples)

        Returns:
            Encoded features (batch, out_channels, time_steps)
        """
        # Layer 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)

        # Layer 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)

        # Layer 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activation(x)

        return x


class TextEncoder(nn.Module):
    """
    Simple text encoder - encodes text tokens to embeddings.

    This is a simplified version of what Tacotron uses.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Character/phoneme embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional LSTM for context
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles this
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            text: Token indices (batch, max_len)
            text_lengths: Actual lengths of each sequence

        Returns:
            Tuple of (encoded, hidden_state)
        """
        # Embed
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        # Pack if lengths provided
        if text_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )

        # Encode
        outputs, (hidden, cell) = self.lstm(embedded)

        # Unpack if needed
        if text_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True
            )

        return outputs, hidden


class SimpleTTSModel(nn.Module):
    """
    A simplified TTS model combining text encoder and decoder.

    This demonstrates:
    - Module composition
    - Attention mechanism basics
    - Autoregressive decoding concept
    """

    def __init__(
        self,
        vocab_size: int = 100,
        n_mels: int = 80,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=hidden_dim,
            hidden_dim=hidden_dim * 2
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )

        # Decoder (simplified - just projects to mel)
        self.mel_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels)
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        target_mel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text: Token indices (batch, max_text_len)
            text_lengths: Text lengths
            target_mel: Target mel spectrogram for teacher forcing

        Returns:
            Predicted mel spectrogram
        """
        # Encode text
        encoded, _ = self.text_encoder(text, text_lengths)

        # Self-attention on encoded text
        attended, _ = self.attention(encoded, encoded, encoded)

        # Decode to mel
        mel_output = self.mel_decoder(attended)

        # Transpose to (batch, n_mels, time)
        mel_output = mel_output.transpose(1, 2)

        return mel_output


# =============================================================================
# 4. Training Loop Components
# =============================================================================

class DummyTTSDataset(Dataset):
    """Dummy dataset for demonstrating training loop."""

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples

        # Generate dummy data
        self.texts = [
            torch.randint(0, 100, (np.random.randint(10, 50),))
            for _ in range(num_samples)
        ]
        self.mels = [
            torch.randn(80, np.random.randint(50, 200))
            for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'mel': self.mels[idx]
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for batching variable-length sequences.

    This is crucial for TTS where text and audio have different lengths.
    """
    texts = [item['text'] for item in batch]
    mels = [item['mel'] for item in batch]

    # Get lengths
    text_lengths = torch.tensor([len(t) for t in texts])
    mel_lengths = torch.tensor([m.shape[1] for m in mels])

    # Pad texts
    max_text_len = text_lengths.max().item()
    texts_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, text in enumerate(texts):
        texts_padded[i, :len(text)] = text

    # Pad mels
    max_mel_len = mel_lengths.max().item()
    mels_padded = torch.zeros(len(batch), 80, max_mel_len)
    for i, mel in enumerate(mels):
        mels_padded[i, :, :mel.shape[1]] = mel

    return {
        'text': texts_padded,
        'text_lengths': text_lengths,
        'mel': mels_padded,
        'mel_lengths': mel_lengths
    }


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Single training step.

    Args:
        model: The TTS model
        batch: Batch of data
        optimizer: Optimizer
        criterion: Loss function
        device: Compute device

    Returns:
        Loss value
    """
    model.train()

    # Move data to device
    text = batch['text'].to(device)
    text_lengths = batch['text_lengths'].to(device)
    mel = batch['mel'].to(device)

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    mel_pred = model(text, text_lengths)

    # Truncate prediction to match target length
    min_len = min(mel_pred.shape[2], mel.shape[2])
    mel_pred = mel_pred[:, :, :min_len]
    mel = mel[:, :, :min_len]

    # Compute loss
    loss = criterion(mel_pred, mel)

    # Backward pass
    loss.backward()

    # Gradient clipping (important for RNNs)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update weights
    optimizer.step()

    return loss.item()


def training_loop_example():
    """Demonstrate a complete training loop."""
    print("\nTraining Loop Example")
    print("-" * 40)

    # Setup
    device = check_device_availability()

    # Create model
    model = SimpleTTSModel(vocab_size=100, n_mels=80, hidden_dim=256)
    model.to(device)

    # Create dataset and dataloader
    dataset = DummyTTSDataset(num_samples=100)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            loss = train_step(model, batch, optimizer, criterion, device)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("Training complete!")


# =============================================================================
# 5. Inference Example
# =============================================================================

@torch.no_grad()
def inference_example(model: nn.Module, device: torch.device):
    """Demonstrate inference mode."""
    model.eval()

    # Dummy input
    text = torch.randint(0, 100, (1, 20)).to(device)
    text_lengths = torch.tensor([20]).to(device)

    # Generate
    mel_output = model(text, text_lengths)

    print(f"\nInference output shape: {mel_output.shape}")
    return mel_output


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Fundamentals for TTS")
    print("=" * 60)

    print("\n1. Tensor Creation")
    tensor_creation_examples()

    print("\n2. Device Check")
    device = check_device_availability()

    print("\n3. Tensor Operations")
    tensor_operations_for_audio()

    print("\n4. Autograd")
    autograd_example()

    print("\n5. Gradient Flow")
    gradient_flow_example()

    print("\n6. Module Example")
    encoder = AudioEncoder()
    dummy_audio = torch.randn(4, 1, 22050)
    encoded = encoder(dummy_audio)
    print(f"Audio encoder output shape: {encoded.shape}")

    print("\n7. Training Loop")
    training_loop_example()
