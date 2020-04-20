import torch.nn as nn
import torch


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

        self.channel_attention_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_channels, in_channels // 8),
            nn.Linear(in_channels // 8, in_channels // 8),
            nn.Linear(in_channels // 8, in_channels)
        )
        self.channel_attention_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Linear(in_channels, in_channels // 8),
            nn.Linear(in_channels // 8, in_channels // 8),
            nn.Linear(in_channels // 8, in_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        channel_attention = nn.functional.sigmoid(self.channel_attention_avg(x) + self.channel_attention_max(x))
        x *= channel_attention

        spatial_attention = torch.cat([x.max(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)], dim=1)
        spatial_attention = self.spatial_attention(spatial_attention)

        x *= spatial_attention

        identity += x

        return identity


def attention_block(in_channels, out_channels) -> list:
    block = [nn.Conv2d(in_channels, out_channels, 3, padding=1),
             nn.LeakyReLU(),
             AttentionModule(out_channels),
             nn.BatchNorm2d(out_channels)]

    return block


def conv_block(in_channels, out_channels, kernel_size=3) -> list:
    block = []
    if kernel_size == 3:
        block += [nn.ZeroPad2d(1)]
    block += [nn.Conv2d(in_channels, out_channels, kernel_size),
             nn.LeakyReLU(),
             nn.BatchNorm2d(out_channels)]

    return block


class ResidualBox(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual_cfg: list):
        super(ResidualBox, self).__init__()
        self.in_conv = nn.Sequential(*conv_block(in_channels, out_channels))

        residual = []

        for channel in residual_cfg:
            residual += conv_block(in_channels, channel)
            in_channels = channel

        residual += attention_block(in_channels, out_channels)

        self.residual = nn.Sequential(*residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.residual(x) + x
        return x
