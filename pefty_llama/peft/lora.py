from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class LoRA(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 ):
        super().__init__()
        self.config = config
        self.peft_config = peft_config

        if input_dim is None:
            input_dim = self.config.dim
        if output_dim is None:
            output_dim = self.config.dim
        self.lora_down = nn.Parameter(torch.randn(input_dim, peft_config.lora_rank, dtype=peft_config.peft_dtype))
        self.lora_up = nn.Parameter(torch.zeros(peft_config.lora_rank, output_dim, dtype=peft_config.peft_dtype))
        self.rank = peft_config.lora_rank
        self.scaling = peft_config.lora_alpha / peft_config.lora_rank

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(self.peft_config.peft_dtype)
        lora_out = torch.einsum("ij,bsi->bsj", (self.lora_down @ self.lora_up), hidden_states) / self.rank
        return (hidden_states + self.scaling * lora_out).to(self.config.dtype)


class LoRAEmbed(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config

        self.lora_down = nn.Parameter(torch.randn(config.vocab_size, peft_config.lora_rank, dtype=peft_config.peft_dtype))
        self.lora_up = nn.Parameter(torch.zeros(peft_config.lora_rank, config.dim, dtype=peft_config.peft_dtype))
        self.rank = peft_config.lora_rank
        self.scaling = peft_config.lora_alpha / peft_config.lora_rank

    def forward(self, input_ids):
        embedding_matrix = self.lora_down @ self.lora_up
        return F.embedding(input_ids, embedding_matrix)
