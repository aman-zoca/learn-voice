"""
Tacotron2 Acoustic Model
========================
Seq2seq model that generates mel spectrograms from text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..text_encoder.encoder import Tacotron2Encoder


class LocationSensitiveAttention(nn.Module):
    """
    Location-sensitive attention mechanism for Tacotron2.

    Uses cumulative attention weights to encourage monotonic alignment.
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
            attention_weights_cat: Prev + cumulative attention (batch, 2, max_time)
            mask: Memory mask (batch, max_time)

        Returns:
            Tuple of (attention_context, attention_weights)
        """
        # Process query
        processed_query = self.query_layer(query.unsqueeze(1))  # (batch, 1, attention_dim)

        # Process location features
        processed_location = self.location_conv(attention_weights_cat)  # (batch, loc_feat, max_time)
        processed_location = processed_location.transpose(1, 2)  # (batch, max_time, loc_feat)
        processed_location = self.location_dense(processed_location)  # (batch, max_time, attention_dim)

        # Compute energies
        energies = self.v(torch.tanh(
            processed_query + processed_memory + processed_location
        ))  # (batch, max_time, 1)
        energies = energies.squeeze(-1)  # (batch, max_time)

        # Apply mask
        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)

        # Softmax to get attention weights
        attention_weights = F.softmax(energies, dim=1)  # (batch, max_time)

        # Compute context
        attention_context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, max_time)
            memory  # (batch, max_time, encoder_dim)
        )
        attention_context = attention_context.squeeze(1)  # (batch, encoder_dim)

        return attention_context, attention_weights


class Prenet(nn.Module):
    """
    Prenet for Tacotron2 decoder.

    Two fully connected layers with dropout (always on, even during inference).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, in_dim) or (batch, seq_len, in_dim)

        Returns:
            Output with same batch dimensions
        """
        # Always apply dropout (even during inference)
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                x = F.dropout(x, p=self.dropout, training=True)
            else:
                x = layer(x)

        return x


class Postnet(nn.Module):
    """
    Postnet for Tacotron2.

    5 convolutional layers that refine the mel spectrogram prediction.
    """

    def __init__(
        self,
        n_mels: int = 80,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5
    ):
        super().__init__()

        self.convolutions = nn.ModuleList()

        # First conv
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    n_mels, postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    padding=(postnet_kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5)
            )
        )

        # Middle convs
        for _ in range(postnet_n_convolutions - 2):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_embedding_dim, postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        padding=(postnet_kernel_size - 1) // 2
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )

        # Last conv
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_embedding_dim, n_mels,
                    kernel_size=postnet_kernel_size,
                    padding=(postnet_kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(n_mels),
                nn.Dropout(0.5)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram (batch, n_mels, time)

        Returns:
            Refined mel spectrogram residual
        """
        for conv in self.convolutions:
            x = conv(x)

        return x


class Tacotron2Decoder(nn.Module):
    """
    Tacotron2 autoregressive decoder.
    """

    def __init__(
        self,
        n_mels: int = 80,
        encoder_dim: int = 512,
        attention_dim: int = 128,
        decoder_dim: int = 1024,
        prenet_dim: int = 256,
        max_decoder_steps: int = 1000,
        gate_threshold: float = 0.5,
        attention_location_features: int = 32,
        attention_location_kernel_size: int = 31
    ):
        super().__init__()

        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold

        # Prenet
        self.prenet = Prenet(n_mels, prenet_dim, prenet_dim)

        # Attention
        self.attention = LocationSensitiveAttention(
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            location_features=attention_location_features,
            location_kernel_size=attention_location_kernel_size
        )

        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(
            prenet_dim + encoder_dim,
            decoder_dim
        )

        # Linear projection
        self.linear_projection = nn.Linear(
            decoder_dim + encoder_dim,
            n_mels
        )

        # Gate (stop token) prediction
        self.gate_layer = nn.Linear(decoder_dim + encoder_dim, 1)

    def _init_decoder_state(
        self,
        memory: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Initialize decoder hidden state."""
        batch_size = memory.size(0)
        max_time = memory.size(1)
        device = memory.device

        # Attention state
        attention_hidden = torch.zeros(batch_size, self.decoder_dim, device=device)
        attention_cell = torch.zeros(batch_size, self.decoder_dim, device=device)

        # Attention weights
        attention_weights = torch.zeros(batch_size, max_time, device=device)
        attention_weights_cum = torch.zeros(batch_size, max_time, device=device)

        # Context
        attention_context = torch.zeros(batch_size, self.encoder_dim, device=device)

        # Processed memory (compute once)
        processed_memory = self.attention.memory_layer(memory)

        return (
            attention_hidden, attention_cell,
            attention_weights, attention_weights_cum,
            attention_context, processed_memory
        )

    def _decode_step(
        self,
        decoder_input: torch.Tensor,
        attention_hidden: torch.Tensor,
        attention_cell: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_weights_cum: torch.Tensor,
        attention_context: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Single decoder step."""
        # Prenet
        prenet_output = self.prenet(decoder_input)

        # Decoder RNN input
        rnn_input = torch.cat([prenet_output, attention_context], dim=-1)

        # Decoder RNN
        attention_hidden, attention_cell = self.decoder_rnn(
            rnn_input, (attention_hidden, attention_cell)
        )

        # Attention
        attention_weights_cat = torch.stack(
            [attention_weights, attention_weights_cum], dim=1
        )

        attention_context, attention_weights = self.attention(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask
        )

        # Update cumulative attention
        attention_weights_cum = attention_weights_cum + attention_weights

        # Project to mel
        decoder_output = torch.cat([attention_hidden, attention_context], dim=-1)
        mel_output = self.linear_projection(decoder_output)

        # Gate prediction
        gate_output = self.gate_layer(decoder_output)

        return (
            mel_output, gate_output,
            attention_hidden, attention_cell,
            attention_weights, attention_weights_cum,
            attention_context
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass (training with teacher forcing).

        Args:
            memory: Encoder outputs (batch, max_time, encoder_dim)
            memory_lengths: Encoder output lengths (batch,)
            targets: Target mel spectrograms (batch, n_mels, target_time)

        Returns:
            Tuple of (mel_outputs, gate_outputs, alignments)
        """
        batch_size = memory.size(0)
        device = memory.device

        # Create mask
        max_len = memory.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0)
        mask = mask >= memory_lengths.unsqueeze(1)

        # Initialize
        (
            attention_hidden, attention_cell,
            attention_weights, attention_weights_cum,
            attention_context, processed_memory
        ) = self._init_decoder_state(memory)

        # Initialize decoder input (go frame)
        decoder_input = torch.zeros(batch_size, self.n_mels, device=device)

        # Prepare targets for teacher forcing
        if targets is not None:
            targets = targets.transpose(1, 2)  # (batch, time, n_mels)

        mel_outputs = []
        gate_outputs = []
        alignments = []

        # Decode
        if targets is not None:
            # Teacher forcing
            for t in range(targets.size(1)):
                (
                    mel_output, gate_output,
                    attention_hidden, attention_cell,
                    attention_weights, attention_weights_cum,
                    attention_context
                ) = self._decode_step(
                    decoder_input,
                    attention_hidden, attention_cell,
                    attention_weights, attention_weights_cum,
                    attention_context,
                    memory, processed_memory, mask
                )

                mel_outputs.append(mel_output)
                gate_outputs.append(gate_output)
                alignments.append(attention_weights)

                # Use target as next input (teacher forcing)
                decoder_input = targets[:, t]
        else:
            # Inference
            for _ in range(self.max_decoder_steps):
                (
                    mel_output, gate_output,
                    attention_hidden, attention_cell,
                    attention_weights, attention_weights_cum,
                    attention_context
                ) = self._decode_step(
                    decoder_input,
                    attention_hidden, attention_cell,
                    attention_weights, attention_weights_cum,
                    attention_context,
                    memory, processed_memory, mask
                )

                mel_outputs.append(mel_output)
                gate_outputs.append(gate_output)
                alignments.append(attention_weights)

                # Check stop condition
                if torch.sigmoid(gate_output).item() > self.gate_threshold:
                    break

                # Use prediction as next input
                decoder_input = mel_output

        # Stack outputs
        mel_outputs = torch.stack(mel_outputs, dim=2)  # (batch, n_mels, time)
        gate_outputs = torch.cat(gate_outputs, dim=1)  # (batch, time)
        alignments = torch.stack(alignments, dim=1)  # (batch, time, max_time)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    """
    Complete Tacotron2 model.

    Combines:
    - Text encoder
    - Attention-based decoder
    - Postnet for mel refinement
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        encoder_embedding_dim: int = 512,
        encoder_hidden_dim: int = 512,
        encoder_n_convolutions: int = 3,
        encoder_kernel_size: int = 5,
        attention_dim: int = 128,
        decoder_dim: int = 1024,
        prenet_dim: int = 256,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 5,
        max_decoder_steps: int = 1000,
        gate_threshold: float = 0.5
    ):
        super().__init__()

        # Encoder
        self.encoder = Tacotron2Encoder(
            vocab_size=vocab_size,
            embedding_dim=encoder_embedding_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            num_conv_layers=encoder_n_convolutions,
            conv_kernel_size=encoder_kernel_size
        )

        # Decoder
        self.decoder = Tacotron2Decoder(
            n_mels=n_mels,
            encoder_dim=encoder_hidden_dim,
            attention_dim=attention_dim,
            decoder_dim=decoder_dim,
            prenet_dim=prenet_dim,
            max_decoder_steps=max_decoder_steps,
            gate_threshold=gate_threshold
        )

        # Postnet
        self.postnet = Postnet(
            n_mels=n_mels,
            postnet_embedding_dim=postnet_embedding_dim,
            postnet_kernel_size=postnet_kernel_size,
            postnet_n_convolutions=postnet_n_convolutions
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text: Token indices (batch, max_text_len)
            text_lengths: Text lengths (batch,)
            mel_targets: Target mels for training (batch, n_mels, time)

        Returns:
            Dictionary with outputs
        """
        # Encode text
        encoder_outputs = self.encoder(text, text_lengths)

        # Decode
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, text_lengths, mel_targets
        )

        # Postnet refinement
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'gate_outputs': gate_outputs,
            'alignments': alignments
        }

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode.

        Args:
            text: Token indices (batch, max_text_len)
            text_lengths: Text lengths (optional)

        Returns:
            Dictionary with outputs
        """
        self.eval()

        if text_lengths is None:
            text_lengths = torch.tensor([text.size(1)], device=text.device)

        return self.forward(text, text_lengths, mel_targets=None)
