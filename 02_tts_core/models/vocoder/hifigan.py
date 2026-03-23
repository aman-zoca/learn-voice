"""
HiFi-GAN Vocoder
================
High-fidelity generative adversarial network for mel-to-waveform synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch.nn.utils import weight_norm, remove_weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate same padding for convolution."""
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    """
    Residual block with dilated convolutions (Type 1).

    Used in HiFi-GAN generator.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5)
    ):
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels, channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=get_padding(kernel_size, dilation)
                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels, channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=get_padding(kernel_size, 1)
                ))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = conv2(x)
            x = x + residual

        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class ResBlock2(nn.Module):
    """
    Residual block with dilated convolutions (Type 2 - simpler).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3)
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        for dilation in dilations:
            self.convs.append(
                weight_norm(nn.Conv1d(
                    channels, channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=get_padding(kernel_size, dilation)
                ))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = x + residual

        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator.

    Converts mel spectrograms to audio waveforms through:
    1. Initial convolution
    2. Upsampling blocks with transposed convolutions
    3. Multi-receptive field fusion (MRF) with residual blocks
    4. Final convolution to audio
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_initial_channel: int = 512,
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5)
        ),
        resblock_type: str = '1'
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = weight_norm(nn.Conv1d(
            in_channels, upsample_initial_channel,
            kernel_size=7, stride=1, padding=3
        ))

        # Upsampling blocks
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))

            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    in_ch, out_ch,
                    kernel_size=k, stride=u,
                    padding=(k - u) // 2
                ))
            )

        # Multi-receptive field fusion blocks
        self.resblocks = nn.ModuleList()

        ResBlock = ResBlock1 if resblock_type == '1' else ResBlock2

        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))

            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        # Final convolution
        self.conv_post = weight_norm(nn.Conv1d(
            ch, 1,
            kernel_size=7, stride=1, padding=3
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram (batch, n_mels, time)

        Returns:
            Audio waveform (batch, 1, time * product(upsample_rates))
        """
        x = self.conv_pre(x)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # Multi-receptive field fusion
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class PeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator component.

    Evaluates audio at different periodic intervals.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False
    ):
        super().__init__()

        self.period = period

        norm_f = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform (batch, 1, time)

        Returns:
            Tuple of (output, feature_maps)
        """
        fmap = []

        # Reshape to 2D for period-based processing
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator component.

    Evaluates audio at different resolutions.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_f = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])

        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform (batch, 1, time)

        Returns:
            Tuple of (output, feature_maps)
        """
        fmap = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator.

    Combines multiple period discriminators with different periods.
    """

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Args:
            y: Real audio (batch, 1, time)
            y_hat: Generated audio (batch, 1, time)

        Returns:
            Tuple of (real_outputs, fake_outputs, real_fmaps, fake_fmaps)
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator.

    Combines scale discriminators at different resolutions.
    """

    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])

        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Args:
            y: Real audio (batch, 1, time)
            y_hat: Generated audio (batch, 1, time)

        Returns:
            Tuple of (real_outputs, fake_outputs, real_fmaps, fake_fmaps)
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HiFiGAN(nn.Module):
    """
    Complete HiFi-GAN model with generator and discriminators.
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5)
        )
    ):
        super().__init__()

        self.generator = HiFiGANGenerator(
            in_channels=in_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes
        )

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate audio from mel spectrogram."""
        return self.generator(mel)

    @torch.no_grad()
    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference mode with weight norm removed."""
        self.generator.eval()
        return self.generator(mel)


# Factory function
def create_hifigan(config: Optional[dict] = None) -> HiFiGAN:
    """Create HiFi-GAN model."""
    if config is None:
        config = {}

    return HiFiGAN(**config)
