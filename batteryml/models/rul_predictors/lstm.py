# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch
import torch.nn as nn

from batteryml.builders import MODELS
from batteryml.models.nn_model import NNModel

class SwitchableActivation(nn.Module):
    """
    Learns to dynamically switch/blend between ReLU and Tanh.
    Formula: Output = σ(α) * ReLU(x) + (1 - σ(α)) * Tanh(x)
    """
    def __init__(self):
        super().__init__()
        # Initializing parameter at 0.0 means sigmoid(0.0) = 0.5 (equal 50/50 mix)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Calculate the gate value between 0.0 and 1.0
        gate = torch.sigmoid(self.alpha)
        # Smoothly blend the two activation transformations
        return gate * self.relu(x) + (1.0 - gate) * self.tanh(x)

    def get_switch_weight(self):
        """Returns the current weight (1.0 = Pure ReLU, 0.0 = Pure Tanh)"""
        return torch.sigmoid(self.alpha).item()

@MODELS.register()
class LSTMRULPredictor(NNModel):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 **kwargs):
        NNModel.__init__(self, **kwargs)
        self.lstm = nn.LSTM(
            in_channels * input_width, channels, 2, batch_first=True)
        self.fc = nn.Linear(channels, 1)

         # Instantiate the learnable switch layer
        self.act_switch = SwitchableActivation()

    def forward(self,
                feature: torch.Tensor,
                label: torch.Tensor,
                return_loss: bool = False):
        if feature.ndim == 3:
            feature = feature.unsqueeze(1)
        B, _, H, _ = feature.size()
        x = feature.permute(0, 2, 1, 3).contiguous().view(B, H, -1)
        x, _ = self.lstm(x)
        x = x[:, -1].contiguous().view(B, -1)
        #x = self.fc(x).view(-1)
        x = self.act_switch(self.fc(x)).view(-1)
        if return_loss:
            return torch.mean((x - label.view(-1)) ** 2)

        return x
