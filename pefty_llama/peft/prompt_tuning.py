import torch
import torch.nn as nn
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class AddSoftPrompt(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(peft_config.num_prefix_tokens, config.dim, dtype=config.dtype))

    def forward(self, hidden_states):
        batch_size, seq_len, dim = hidden_states.shape
        soft_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([soft_prompt, hidden_states], dim=1)
