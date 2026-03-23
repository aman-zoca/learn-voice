"""
Neural Network Fundamentals for TTS
===================================
This module covers essential neural network concepts used in TTS systems.

Key Concepts:
- Feed-forward networks
- Convolutional networks for sequence modeling
- Recurrent networks (LSTM, GRU)
- Attention mechanisms
- Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# =============================================================================
# 1. Basic Building Blocks
# =============================================================================

class LinearWithActivation(nn.Module):
    """
    Linear layer with optional activation and normalization.

    This is the most basic building block - used everywhere in TTS.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = 'relu',
        use_norm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.use_norm = use_norm
        self.dropout = nn.Dropout(dropout)

        if use_norm:
            self.norm = nn.LayerNorm(out_features)

        # Activation selection
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity()
        }
        self.activation = activations.get(activation, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


# =============================================================================
# 2. Convolutional Networks for Audio
# =============================================================================

class Conv1dBlock(nn.Module):
    """
    1D Convolutional block with normalization and activation.

    Used in:
    - Text encoders (processing character sequences)
    - Audio encoders (processing mel spectrograms)
    - Vocoders (generating audio waveforms)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        activation: str = 'relu',
        norm_type: str = 'batch'
    ):
        super().__init__()

        if padding is None:
            # Same padding
            padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        # Normalization
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'none': nn.Identity()
        }
        self.activation = activations.get(activation, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, length)

        Returns:
            Output tensor (batch, out_channels, length)
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Skip connections help training deep networks by:
    - Enabling gradient flow
    - Allowing the network to learn incremental changes
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()

        self.conv1 = Conv1dBlock(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual  # Skip connection


class DilatedConvStack(nn.Module):
    """
    Stack of dilated convolutions with increasing dilation rates.

    Used in WaveNet and HiFi-GAN for large receptive fields.
    Dilation pattern: 1, 2, 4, 8, 16, ... (exponential growth)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        num_layers: int = 4,
        dilation_base: int = 2
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ResidualBlock(
                channels,
                kernel_size=kernel_size,
                dilation=dilation_base ** i
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# 3. Recurrent Networks
# =============================================================================

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence encoding.

    Used in:
    - Text encoders (Tacotron2)
    - Speaker encoders
    - Prosody modeling
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output is 2 * hidden_size due to bidirectional
        self.output_size = hidden_size * 2

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
            lengths: Sequence lengths for packing

        Returns:
            Tuple of (outputs, (hidden, cell))
        """
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

        outputs, (hidden, cell) = self.lstm(x)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True
            )

        return outputs, (hidden, cell)


class AutoregressiveLSTM(nn.Module):
    """
    Autoregressive LSTM decoder.

    Used in Tacotron2's decoder - generates one frame at a time,
    using the previous output as input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single step forward pass.

        Args:
            x: Input (batch, 1, input_size)
            hidden: Previous hidden state

        Returns:
            Tuple of (output, (hidden, cell))
        """
        return self.lstm(x, hidden)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state."""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )


# =============================================================================
# 4. Attention Mechanisms
# =============================================================================

class DotProductAttention(nn.Module):
    """
    Simple dot-product attention.

    Attention allows the model to focus on relevant parts of the input
    when generating each output.

    score = Q · K^T / sqrt(d_k)
    attention = softmax(score) · V
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, query_len, d_model)
            key: (batch, key_len, d_model)
            value: (batch, key_len, d_model)
            mask: Optional mask (batch, query_len, key_len)

        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class LocationSensitiveAttention(nn.Module):
    """
    Location-sensitive attention used in Tacotron2.

    This attention mechanism considers previous attention weights
    to encourage monotonic alignment (left-to-right reading).
    """

    def __init__(
        self,
        attention_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        location_features: int = 32,
        location_kernel_size: int = 31
    ):
        super().__init__()

        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        # Location features
        self.location_conv = nn.Conv1d(
            2, location_features,
            kernel_size=location_kernel_size,
            padding=(location_kernel_size - 1) // 2,
            bias=False
        )
        self.location_dense = nn.Linear(location_features, attention_dim, bias=False)

        self.score_mask_value = float('-inf')

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        attention_weights_cat: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Decoder state (batch, decoder_dim)
            memory: Encoder outputs (batch, max_time, encoder_dim)
            processed_memory: Pre-processed memory (batch, max_time, attention_dim)
            attention_weights_cat: Previous attention weights (batch, 2, max_time)
            mask: Memory mask

        Returns:
            Tuple of (attention_context, attention_weights)
        """
        # Process query
        processed_query = self.query_layer(query.unsqueeze(1))

        # Process location features
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)

        # Compute energies
        energies = self.v(torch.tanh(
            processed_query + processed_memory + processed_attention
        ))
        energies = energies.squeeze(-1)

        # Apply mask
        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)

        # Attention weights
        attention_weights = F.softmax(energies, dim=-1)

        # Compute context
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as used in Transformers.

    Multiple attention heads allow the model to attend to
    different parts of the sequence for different reasons.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, query_len, d_model)
            key: (batch, key_len, d_model)
            value: (batch, key_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(context)

        return output, attention_weights


# =============================================================================
# 5. Transformer Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    Adds position information to the embeddings since transformers
    have no inherent notion of sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Position-encoded tensor
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network from Transformer.

    Two linear transformations with activation in between.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Consists of:
    1. Multi-head self-attention
    2. Feed-forward network
    Both with residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Encoded output
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class TransformerEncoder(nn.Module):
    """
    Full transformer encoder stack.

    Used in modern TTS systems like FastSpeech and VITS.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Encoded sequence
        """
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# =============================================================================
# 6. Example Usage and Tests
# =============================================================================

def test_attention():
    """Test attention mechanisms."""
    print("Testing Attention Mechanisms")
    print("-" * 40)

    batch_size = 4
    seq_len = 100
    d_model = 256

    # Input
    x = torch.randn(batch_size, seq_len, d_model)

    # Dot product attention
    dot_attn = DotProductAttention(d_model)
    output, weights = dot_attn(x, x, x)
    print(f"Dot attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Multi-head attention
    mha = MultiHeadAttention(d_model, num_heads=8)
    output, weights = mha(x, x, x)
    print(f"Multi-head attention output shape: {output.shape}")


def test_transformer():
    """Test transformer encoder."""
    print("\nTesting Transformer Encoder")
    print("-" * 40)

    batch_size = 4
    seq_len = 50
    d_model = 256

    # Input
    x = torch.randn(batch_size, seq_len, d_model)

    # Transformer encoder
    encoder = TransformerEncoder(
        d_model=d_model,
        num_layers=4,
        num_heads=8,
        d_ff=1024
    )

    output = encoder(x)
    print(f"Transformer output shape: {output.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Number of parameters: {num_params:,}")


def test_convolutions():
    """Test convolutional blocks."""
    print("\nTesting Convolutional Blocks")
    print("-" * 40)

    batch_size = 4
    channels = 256
    length = 100

    # Input
    x = torch.randn(batch_size, channels, length)

    # Conv block
    conv_block = Conv1dBlock(channels, channels, kernel_size=3)
    output = conv_block(x)
    print(f"Conv block output shape: {output.shape}")

    # Residual block
    res_block = ResidualBlock(channels)
    output = res_block(x)
    print(f"Residual block output shape: {output.shape}")

    # Dilated conv stack
    dilated_stack = DilatedConvStack(channels, num_layers=4)
    output = dilated_stack(x)
    print(f"Dilated stack output shape: {output.shape}")


if __name__ == "__main__":
    test_attention()
    test_transformer()
    test_convolutions()
