import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pefty_llama.modeling import LLaMAModel, LLaMAConfig, NoInitLinear, NoInit8bitLinear, RotaryEmbedding, apply_rotary_pos_emb, check_nan


class IA3Attention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        if config.use_8bit:
            self.q_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.k_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.v_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.o_proj = NoInit8bitLinear(config.dim, config.dim, bias=False, threshold=6.0, has_fp16_weights=False)
        else:
            self.q_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.k_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.v_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
            self.o_proj = NoInitLinear(config.dim, config.dim, bias=False, dtype=config.dtype)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

        # IA3-specific parameters:
        self.peft_l_k = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim, dtype=config.dtype))
        self.peft_l_v = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim, dtype=config.dtype))

    def forward(self, hidden_states, attention_mask, cos, sin, kv_cache=None):
        """
        precomputed_kv_hidden_states is for init (pre-compute KV activations, e.g. for added prefixes)
        kv_cache is for generation (cached past KV)
        """
        batch_size, q_seq_len, hidden_dim = hidden_states.size()

        # (batch_size, num_heads, q_seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos=cos, sin=sin)
        if kv_cache:
            key_states = torch.cat([kv_cache["key"], key_states], dim=2)
            value_states = torch.cat([kv_cache["value"], value_states], dim=2)

        # IA3-specific:
        query_states = query_states * self.peft_l_k
        value_states = value_states * self.peft_l_v
        # end of IA3-specific

        scores = torch.matmul(
            query_states, key_states.transpose(3, 2).type_as(query_states) / math.sqrt(self.head_dim)
        )
        scores += attention_mask

        # (batch_size, num_heads, q_seq_len, kv_seq_len)
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        # (batch_size, num_heads, q_seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states.type_as(query_states))
        # (batch_size, q_seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, hidden_dim,
        )
        attn_output = self.o_proj(attn_output)
        check_nan(attn_output)
        if kv_cache:
            new_kv_cache = {"key": key_states, "value": value_states}
            return {"attn_output": attn_output, "kv_cache": new_kv_cache}
        else:
            return {"attn_output": attn_output}


class IA3MLP(nn.Module):
    def __init__(
        self,
        config: LLaMAConfig,
        multiple_of: int = 256,
    ):
        super().__init__()
        dim = config.dim
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if config.use_8bit:
            self.gate_proj = NoInit8bitLinear(dim, hidden_dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.up_proj = NoInit8bitLinear(dim, hidden_dim, bias=False, threshold=6.0, has_fp16_weights=False)
            self.down_proj = NoInit8bitLinear(hidden_dim, dim, bias=False, threshold=6.0, has_fp16_weights=False)
        else:
            self.gate_proj = NoInitLinear(dim, hidden_dim, bias=False, dtype=config.dtype)
            self.up_proj = NoInitLinear(dim, hidden_dim, bias=False, dtype=config.dtype)
            self.down_proj = NoInitLinear(hidden_dim, dim, bias=False, dtype=config.dtype)

        # IA3-specific parameters:
        self.peft_l_ffn = nn.Parameter(torch.ones(1, 1, hidden_dim, dtype=config.dtype))

    def forward(self, x):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        # IA3-specific:
        h = h * self.peft_l_ffn
        # end of IA3-specific
        return self.down_proj(h)


class IA3(nn.Module):
    def __init__(self, model: LLaMAModel):
        super().__init__()
        self.base_model = model
        model_config = model.config

        for layer in self.base_model.model.layers:
            # you also need to copy the parameters of the layer to the new layer
            patched_attn = IA3Attention(model_config)
            current_attn = layer.self_attn
            patched_attn.q_proj.weight = current_attn.q_proj.weight
            patched_attn.k_proj.weight = current_attn.k_proj.weight
            patched_attn.v_proj.weight = current_attn.v_proj.weight
            patched_attn.o_proj.weight = current_attn.o_proj.weight
            patched_attn.rotary_emb = current_attn.rotary_emb

            layer.self_attn = patched_attn
            del current_attn

            patched_mlp = IA3MLP(model_config)
            current_mlp = layer.mlp
            patched_mlp.gate_proj.weight = current_mlp.gate_proj.weight
            patched_mlp.up_proj.weight = current_mlp.up_proj.weight
            patched_mlp.down_proj.weight = current_mlp.down_proj.weight
            
            layer.mlp = patched_mlp
            del current_mlp

        # cleanup memory freed by deleting the old layers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        for name, param in self.base_model.named_parameters():
            if "peft_" in name: continue
            param.requires_grad = False

        # monkey patch the methods
        self.forward = self.base_model.forward
        self.generate = self.base_model.generate


class IA3ForAttn(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.peft_l_k = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim, dtype=config.dtype))
        self.peft_l_v = nn.Parameter(torch.ones(1, self.n_heads, 1, self.head_dim, dtype=config.dtype))

    def forward(self, key_states, value_states):
        return key_states * self.peft_l_k, value_states * self.peft_l_v


class IA3ForMLP(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        multiple_of = 256
        intermediate_dim = 4 * config.dim
        intermediate_dim = int(2 * intermediate_dim / 3)
        intermediate_dim = multiple_of * ((intermediate_dim + multiple_of - 1) // multiple_of)

        self.peft_l_ffn = nn.Parameter(torch.ones(1, 1, intermediate_dim, dtype=config.dtype))

    def forward(self, intermediate_state):
        return self.peft_l_ffn * intermediate_state
