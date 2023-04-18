import torch
import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class PrefixAdapter(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        # "batch_size"=1, num_heads, num_prefix_tokens, head_dim
        self.prefix_k = nn.Parameter(torch.randn(
            1, config.n_heads, peft_config.num_prefix_tokens, config.head_dim, dtype=config.dtype))
        self.prefix_v = nn.Parameter(torch.randn(
            1, config.n_heads, peft_config.num_prefix_tokens, config.head_dim, dtype=config.dtype))
        self.gate = nn.Parameter(torch.zeros(1, config.n_heads, 1, 1))

    def forward(self, query_states):
        batch_size, q_seq_len, dim = query_states.shape
        # "batch_size"=1, num_heads, num_prefix_tokens, head_dim
        prefix_k = self.prefix_k.expand(batch_size, -1, -1, -1)
        prefix_v = self.prefix_v.expand(batch_size, -1, -1, -1)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=prefix_k,
            value=prefix_v,
        )
        return F.tanh(self.gate) * attn_output
