# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import torch
from torch import nn


def build_mlp(
    *,
    n_in,
    n_hidden,
    n_layers,
    out_size=None,
    act=None,
    use_layer_norm=False,
    dropout=0,
):
    if act is None:
        act = GELU()
    build_norm_layer = (
        lambda: nn.LayerNorm(n_hidden) if use_layer_norm else nn.Sequential()
    )
    build_dropout_layer = (
        lambda: nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
    )

    last_size = n_in
    vals_net = []
    for _ in range(n_layers):
        vals_net.extend(
            [
                nn.Linear(last_size, n_hidden),
                build_norm_layer(),
                act,
                build_dropout_layer(),
            ]
        )
        last_size = n_hidden
    if out_size is not None:
        vals_net.append(nn.Linear(last_size, out_size))
    return nn.Sequential(*vals_net)


def input_size(num_faces, num_dice):
    return 1 + 1 + (2 * num_faces * num_dice + 1) + 2 * output_size(num_faces, num_dice)


def output_size(num_faces, num_dice):
    return num_faces ** num_dice


class Net2(nn.Module):
    def __init__(
        self,
        *,
        num_faces,
        num_dice,
        n_hidden=256,
        use_layer_norm=False,
        dropout=0,
        n_layers=3,
    ):
        super().__init__()

        n_in = input_size(num_faces, num_dice)
        self.body = build_mlp(
            n_in=n_in,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.output = nn.Linear(
            n_hidden if n_layers > 0 else n_in, output_size(num_faces, num_dice)
        )
        # Make initial predictions closer to 0.
        with torch.no_grad():
            self.output.weight.data *= 0.01
            self.output.bias *= 0.01

    def forward(self, packed_input: torch.Tensor):
        return self.output(self.body(packed_input))


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)
