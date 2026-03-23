"""
Text Encoder Module
===================
Encodes text/phoneme sequences into hidden representations for TTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

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


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and activation.

    Used in the Tacotron2 encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.5
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Tacotron2Encoder(nn.Module):
    """
    Tacotron2 text encoder.

    Architecture:
    - Character/phoneme embedding
    - 3 convolutional layers
    - Bidirectional LSTM
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        encoder_hidden_dim: int = 512,
        num_conv_layers: int = 3,
        conv_kernel_size: int = 5,
        conv_dropout: float = 0.5,
        lstm_hidden_dim: int = 512,
        lstm_dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Convolutional layers
        conv_channels = encoder_hidden_dim
        self.convolutions = nn.ModuleList()

        for i in range(num_conv_layers):
            in_channels = embedding_dim if i == 0 else conv_channels
            self.convolutions.append(
                ConvBlock(
                    in_channels, conv_channels,
                    kernel_size=conv_kernel_size,
                    dropout=conv_dropout
                )
            )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text: Token indices (batch, max_text_len)
            text_lengths: Actual text lengths (batch,)

        Returns:
            Encoded text (batch, max_text_len, encoder_hidden_dim)
        """
        # Embedding
        x = self.embedding(text)  # (batch, seq_len, embedding_dim)

        # Conv layers expect (batch, channels, seq_len)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = conv(x)

        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # LSTM
        if text_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, text_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )

        x, _ = self.lstm(x)

        if text_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x


class TransformerTextEncoder(nn.Module):
    """
    Transformer-based text encoder.

    Used in modern TTS systems like FastSpeech and VITS.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text: Token indices (batch, max_text_len)
            text_lengths: Actual text lengths (batch,)

        Returns:
            Encoded text (batch, max_text_len, hidden_dim)
        """
        # Embedding + positional encoding
        x = self.embedding(text)
        x = self.pos_encoding(x)

        # Create attention mask
        if text_lengths is not None:
            max_len = text.size(1)
            mask = torch.arange(max_len, device=text.device).unsqueeze(0)
            mask = mask >= text_lengths.unsqueeze(1)
        else:
            mask = None

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)

        # Output projection
        x = self.output_proj(x)

        return x


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding as used in VITS.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, hidden_dim)

        Returns:
            Position-encoded tensor
        """
        # Conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer block as used in FastSpeech.

    Consists of multi-head attention and position-wise FFN.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 2,
        ffn_dim: int = 1024,
        kernel_size: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Position-wise FFN with conv
        self.ffn = nn.Sequential(
            nn.Conv1d(hidden_dim, ffn_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, hidden_dim)
            mask: Attention mask

        Returns:
            Output tensor
        """
        # Self-attention with residual
        residual = x
        x = self.attn_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.attn_dropout(x)
        x = x + residual

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = x.transpose(1, 2)  # (batch, hidden, seq_len)
        x = self.ffn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden)
        x = x + residual

        return x


class FastSpeechEncoder(nn.Module):
    """
    FastSpeech-style encoder with FFT blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        ffn_dim: int = 1024,
        ffn_kernel_size: int = 9,
        dropout: float = 0.1,
        max_len: int = 2000
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)

        self.layers = nn.ModuleList([
            FFTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                kernel_size=ffn_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text: Token indices (batch, max_text_len)
            text_lengths: Actual text lengths

        Returns:
            Encoded text (batch, max_text_len, hidden_dim)
        """
        # Create mask
        if text_lengths is not None:
            max_len = text.size(1)
            mask = torch.arange(max_len, device=text.device).unsqueeze(0)
            mask = mask >= text_lengths.unsqueeze(1)
        else:
            mask = None

        # Embedding + positional
        x = self.embedding(text)
        x = self.pos_encoding(x)

        # FFT blocks
        for layer in self.layers:
            x = layer(x, mask)

        return x


# Factory function
def create_text_encoder(
    encoder_type: str,
    vocab_size: int,
    **kwargs
) -> nn.Module:
    """
    Create a text encoder.

    Args:
        encoder_type: 'tacotron2', 'transformer', or 'fastspeech'
        vocab_size: Size of vocabulary
        **kwargs: Additional encoder arguments

    Returns:
        Text encoder module
    """
    if encoder_type == 'tacotron2':
        return Tacotron2Encoder(vocab_size, **kwargs)
    elif encoder_type == 'transformer':
        return TransformerTextEncoder(vocab_size, **kwargs)
    elif encoder_type == 'fastspeech':
        return FastSpeechEncoder(vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
