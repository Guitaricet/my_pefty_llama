import torch
import torch.nn as nn


class BitFitAddBias(nn.Module):
    def __init__(self, dim: int, dtype=torch.float16):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, hidden_state):
        return hidden_state + self.bias
