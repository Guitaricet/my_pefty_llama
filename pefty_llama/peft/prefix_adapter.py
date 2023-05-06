import torch
import torch.nn as nn
import torch.nn.functional as F
from pefty_llama.configuration import LLaMAConfig
from .configuration import PeftConfig


class PrefixAdapter(nn.Module):
    def __init__(self, config: LLaMAConfig, peft_config: PeftConfig):
        super().__init__()
        self.config = config
        self.peft_config = peft_config
        # "batch_size"=1, num_heads, num_prefix_tokens, head_dim
        self.prefix_k = nn.Parameter(torch.randn(
            1, config.n_heads, peft_config.num_prefix_tokens, config.head_dim, dtype=peft_config.peft_dtype))
        self.prefix_v = nn.Parameter(torch.randn(
            1, config.n_heads, peft_config.num_prefix_tokens, config.head_dim, dtype=peft_config.peft_dtype))
        self.gate = nn.Parameter(torch.zeros(1, config.n_heads, 1, 1))

    def forward(self, query_states):
        batch_size, num_heads, q_seq_len, head_dim = query_states.shape
        # "batch_size"=1, num_heads, num_prefix_tokens, head_dim
        prefix_k = self.prefix_k.expand(batch_size, -1, -1, -1)
        prefix_v = self.prefix_v.expand(batch_size, -1, -1, -1)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states.to(self.peft_config.peft_dtype),
            key=prefix_k,
            value=prefix_v,
        )
        return (F.tanh(self.gate) * attn_output).to(self.config.dtype)
