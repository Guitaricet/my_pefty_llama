import torch
import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class Adapter(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.down_proj = nn.Linear(config.dim, peft_config.adapter_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.dim, peft_config.adapter_hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.up_proj(F.gelu(self.down_proj(hidden_states))) + hidden_states
