"""
VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
=====================================================================================
End-to-end TTS model combining VAE, normalizing flows, and adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class LayerNorm(nn.Module):
    """Layer normalization for channels-last format."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, time)"""
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WN(nn.Module):
    """
    WaveNet-style dilated convolution stack.

    Used in the posterior encoder and decoder.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        gin_channels: int = 0,
        p_dropout: float = 0.0
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2

            self.in_layers.append(
                nn.Conv1d(
                    hidden_channels, 2 * hidden_channels,
                    kernel_size, dilation=dilation, padding=padding
                )
            )

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            self.res_skip_layers.append(
                nn.Conv1d(hidden_channels, res_skip_channels, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, hidden_channels, time)
            x_mask: Mask (batch, 1, time)
            g: Global conditioning (batch, gin_channels, 1)

        Returns:
            Output tensor
        """
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            # Gated activation
            acts = torch.tanh(x_in[:, :self.hidden_channels]) * \
                   torch.sigmoid(x_in[:, self.hidden_channels:])
            acts = self.drop(acts)

            res_skip = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                res = res_skip[:, :self.hidden_channels]
                skip = res_skip[:, self.hidden_channels:]
                x = (x + res) * x_mask
                output = output + skip
            else:
                output = output + res_skip

        return output * x_mask


class TextEncoder(nn.Module):
    """
    Text encoder for VITS.

    Uses transformer blocks with relative positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        out_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(vocab_size, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Token indices (batch, text_len)
            x_lengths: Text lengths

        Returns:
            Tuple of (x, m, logs, x_mask)
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x.transpose(1, 2)  # (batch, hidden, text_len)

        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).float()

        x = self.encoder(x * x_mask, x_mask)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask


class Encoder(nn.Module):
    """
    Transformer encoder with FFT blocks.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for _ in range(n_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout, batch_first=True)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, filter_channels, kernel_size, p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (batch, hidden, time)
            x_mask: Mask (batch, 1, time)

        Returns:
            Encoded output
        """
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1,
            self.ffn_layers, self.norm_layers_2
        ):
            # Attention
            x_t = x.transpose(1, 2)  # (batch, time, hidden)
            attn_mask = ~x_mask.squeeze(1).bool()
            attn_out, _ = attn(x_t, x_t, x_t, key_padding_mask=attn_mask)
            attn_out = attn_out.transpose(1, 2)  # (batch, hidden, time)

            x = norm1(x + self.drop(attn_out))

            # FFN
            x = norm2(x + ffn(x, x_mask))

        return x * x_mask


class FFN(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0
    ):
        super().__init__()

        self.conv_1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.conv_2 = nn.Conv1d(
            out_channels, in_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class PosteriorEncoder(nn.Module):
    """
    Posterior encoder that encodes mel spectrograms to latent space.
    """

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 0
    ):
        super().__init__()

        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, kernel_size, dilation_rate,
            n_layers, gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Mel spectrogram (batch, n_mels, time)
            x_lengths: Mel lengths
            g: Global conditioning

        Returns:
            Tuple of (z, m, logs, x_mask)
        """
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).float()

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    """
    Residual coupling layer for normalizing flows.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0
    ):
        super().__init__()

        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size,
                    dilation_rate, n_layers, gin_channels
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g, reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g, reverse)

        return x


class ResidualCouplingLayer(nn.Module):
    """Single residual coupling layer."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False
    ):
        super().__init__()

        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, kernel_size, dilation_rate,
            n_layers, gin_channels
        )
        self.post = nn.Conv1d(
            hidden_channels,
            self.half_channels * (2 - mean_only),
            1
        )
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs, dim=[1, 2])
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = None

        return x, logdet


class Flip(nn.Module):
    """Flip layer for flows."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, None]:
        x = torch.flip(x, dims=[1])
        return x, None


class Generator(nn.Module):
    """
    HiFi-GAN style generator for VITS.
    """

    def __init__(
        self,
        initial_channel: int = 192,
        resblock_type: str = '1',
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5)
        ),
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        gin_channels: int = 0
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, 3
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, (k - u) // 2
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, 3, bias=False)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class ResBlock(nn.Module):
    """Residual block for generator."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5)
    ):
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size, 1, padding, d))
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, 1, kernel_size // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class VITS(nn.Module):
    """
    Complete VITS model.

    End-to-end TTS that combines:
    - Text encoder with prior distribution
    - Posterior encoder from mel spectrograms
    - Normalizing flows for distribution matching
    - HiFi-GAN style decoder
    """

    def __init__(
        self,
        vocab_size: int,
        spec_channels: int = 80,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock_type: str = '1',
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5)
        ),
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        n_speakers: int = 0,
        gin_channels: int = 0
    ):
        super().__init__()

        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        # Text encoder
        self.enc_p = TextEncoder(
            vocab_size, inter_channels, hidden_channels,
            filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        # Posterior encoder
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels,
            5, 1, 16, gin_channels
        )

        # Flow
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        # Decoder
        self.dec = Generator(
            inter_channels, resblock_type, resblock_kernel_sizes,
            resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
            upsample_kernel_sizes, gin_channels
        )

        # Speaker embedding
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            x: Text tokens (batch, text_len)
            x_lengths: Text lengths
            y: Mel spectrogram (batch, n_mels, mel_len)
            y_lengths: Mel lengths
            sid: Speaker IDs (batch,)

        Returns:
            Dictionary with outputs
        """
        # Speaker embedding
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # Text encoder
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Posterior encoder
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g)

        # Flow (posterior to prior)
        z_p = self.flow(z, y_mask, g)

        # Decoder
        o = self.dec(z * y_mask, g)

        return {
            'o': o,
            'z': z,
            'z_p': z_p,
            'm_p': m_p,
            'logs_p': logs_p,
            'm_q': m_q,
            'logs_q': logs_q,
            'x_mask': x_mask,
            'y_mask': y_mask
        }

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Inference.

        Args:
            x: Text tokens
            x_lengths: Text lengths
            sid: Speaker ID
            noise_scale: Noise scale for sampling
            length_scale: Duration scale

        Returns:
            Generated audio waveform
        """
        # Speaker embedding
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # Text encoder
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Flow (prior to posterior)
        z = self.flow(z_p, x_mask, g, reverse=True)

        # Decoder
        o = self.dec(z * x_mask, g)

        return o


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create sequence mask from lengths."""
    if max_len is None:
        max_len = lengths.max()

    x = torch.arange(max_len, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


# Factory function
def create_vits(config: Optional[dict] = None) -> VITS:
    """Create VITS model."""
    if config is None:
        config = {}

    return VITS(**config)
