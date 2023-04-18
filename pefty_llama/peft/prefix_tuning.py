import torch
import torch.nn as nn
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class SoftPrefixes(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        if self.peft_config.prefix_use_mlp:
            if self.peft_config.prefix_mlp_intermediate_size is not None:
                intermediate_size = self.peft_config.prefix_mlp_intermediate_size
            else:
                intermediate_size = self.config.dim

            self.initial = nn.Parameter(torch.randn(peft_config.num_prefix_tokens, config.dim, dtype=config.dtype))
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(config.dim, intermediate_size),
                torch.nn.Tanh(),
                torch.nn.Linear(intermediate_size, config.n_layers * 2 * config.dim),
            )
        else:
            self.soft_prompt = nn.Parameter(torch.randn(
                peft_config.num_prefix_tokens, config.n_layers * 2 * config.dim,
                dtype=config.dtype,
            ))

    def forward(self, batch_size):
        if self.peft_config.prefix_use_mlp:
            out = self.mlp(self.initial)
        else:
            out = self.embedding
        # layers, k/v, num_prefix_tokens, dim
        out = out.view(self.peft_config.num_prefix_tokens, self.config.n_layers, 2, self.config.dim)
        return [
            {
                "key": out[:, layer, 0, :].unsqueeze(0).expand(batch_size, -1, -1),
                "value": out[:, layer, 1, :].unsqueeze(0).expand(batch_size, -1, -1),
            }
            for layer in range(self.config.n_layers)
        ]
