import torch
import torch.nn as nn
from .configuration import PeftConfig


class BitFitAddBias(nn.Module):
    def __init__(self, dim: int, peft_config: PeftConfig):
        super().__init__()
        self.peft_config = peft_config
        self.bias = nn.Parameter(torch.zeros(dim, dtype=peft_config.peft_dtype))

    def forward(self, hidden_state):
        input_dtype = hidden_state.dtype
        return (hidden_state.to(self.peft_config.peft_dtype) + self.bias).to(input_dtype)
