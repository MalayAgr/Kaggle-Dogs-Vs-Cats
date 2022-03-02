from __future__ import annotations

from typing import Any

import torch
from torch import nn

import config


class CatsDogsModel(nn.Module):
    op_activs = {
        "sigmoid": nn.Sigmoid,
        "softmax": nn.Softmax,
    }

    def __init__(
        self,
        conv_out_channels: list[int],
        kernel_sizes: list[int],
        linear_out_features: list[int],
        n_targets: int = 1,
        in_channels: int = 3,
        dropout: float = 0.0,
        op_activation: str = "sigmoid",
    ) -> None:
        super().__init__()

        layers = []
        in_c = in_channels

        for out_channels, kernel_size in zip(conv_out_channels, kernel_sizes):
            conv_block = self._make_conv_block(
                in_channels=in_c,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
            )
            layers.append(conv_block)
            in_c = out_channels

        layers.append(nn.Flatten())

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        in_features = self._get_linear_in_features(layers, in_channels)

        for out_features in linear_out_features:
            linear_block = self._make_linear_block(
                in_features=in_features, out_features=out_features
            )
            layers.append(linear_block)
            in_features = out_features

        layers.append(nn.Linear(in_features, n_targets))
        layers.append(self.op_activs[op_activation]())
        self.layers = nn.Sequential(*layers)

    def _make_conv_block(
        self, in_channels: int, out_channels: int, kernel_size: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

    def _get_linear_in_features(
        self, layers: list[nn.Module], in_channels: int = 3
    ) -> int:
        """
        Automatically calculate the in_features for the linear layer
        Immediately after the final convolutional layer.
        """
        x = torch.rand(
            1,
            in_channels,
            config.IMG_HEIGHT,
            config.IMG_WIDTH,
            dtype=torch.float32,
            device=config.DEVICE,
        )
        m = nn.Sequential(*layers)
        m.to(config.DEVICE)

        return m(x).size(-1)

    def _make_linear_block(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    def init_weights(self) -> None:
        def xavier(layer: nn.Module):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)

        self.apply(xavier)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image).squeeze(1)


def get_model_params(params: dict[str, Any]) -> dict[str, Any]:
    dict_params = {}

    dict_params["conv_out_channels"] = [
        params["conv1_out"],
        params["conv2_out"],
        params["conv3_out"],
        params["conv4_out"],
    ]

    k1, k2 = params["kernel_size1"], params["kernel_size2"]

    dict_params["kernel_sizes"] = [k1, k1, k2, k2]

    dict_params["linear_out_features"] = [
        params["linear1_out"],
        params["linear2_out"],
        params["linear3_out"],
    ]

    dict_params["dropout"] = params["dropout"]

    return dict_params
