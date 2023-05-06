import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class Adapter(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        self.down_proj = nn.Linear(
            config.dim, peft_config.adapter_hidden_size, bias=False,
            dtype=peft_config.peft_dtype,
        )
        self.up_proj = nn.Linear(
            peft_config.adapter_hidden_size, config.dim, bias=False,
            dtype=peft_config.peft_dtype,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(self.peft_config.peft_dtype)
        out = self.up_proj(F.gelu(self.down_proj(hidden_states))) + hidden_states
        return out.to(self.config.dtype)
