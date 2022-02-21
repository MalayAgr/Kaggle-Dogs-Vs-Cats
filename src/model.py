from __future__ import annotations

import torch
from torch import nn

import config


class CatsDogsModel(nn.Module):
    def __init__(
        self,
        conv_out_channels: list[int],
        kernel_sizes: list[int],
        linear_out_features: list[int],
        n_targets: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = []
        in_channels = 3

        for out_channels, kernel_size in zip(conv_out_channels, kernel_sizes):
            conv_block = self._make_conv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
            )
            layers.append(conv_block)
            in_channels = out_channels

        layers.append(nn.Flatten())

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        in_features = self._get_linear_in_features(layers)

        for out_features in linear_out_features:
            linear_block = self._make_linear_block(
                in_features=in_features, out_features=out_features
            )
            layers.append(linear_block)
            in_features = out_features

        layers.append(nn.Linear(in_features, n_targets))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

    def _get_linear_in_features(self, layers: list[nn.Module]):
        x = torch.rand(1, 3, config.IMG_HEIGHT, config.IMG_WIDTH, dtype=torch.float32)
        x = x.to(config.DEVICE)
        m = nn.Sequential(*layers)
        m.to(config.DEVICE)

        return m(x).size(-1)

    def _make_linear_block(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image).squeeze(1)
