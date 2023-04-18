import torch
import torch.nn as nn
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class LoRA(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.lora_down = nn.Parameter(torch.randn(config.dim, peft_config.lora_rank, dtype=config.dtype))
        self.lora_up = nn.Parameter(torch.zeros(peft_config.lora_rank, config.dim, dtype=config.dtype))
        self.rank = peft_config.lora_rank
        self.scaling = peft_config.lora_alpha / peft_config.lora_rank

    def forward(self, hidden_states):
        lora_out = torch.einsum("ij,bsi->bsj", (self.lora_down @ self.lora_up), hidden_states) / self.rank
        return hidden_states + self.scaling * lora_out
