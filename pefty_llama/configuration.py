from typing import Any
import dataclasses
import torch


@dataclasses.dataclass
class LLaMAConfig:
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int = 32000
    max_seq_length: int = 2048
    dtype: Any = torch.float16
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_8bit: bool = False
    gradient_checkpointing: bool = False

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    def to_dict(self):
        return dataclasses.asdict(self)


LLAMA_7B_CONFIG = LLaMAConfig(
    dim=4096,
    n_layers=32,
    n_heads=32,
)
DEBUG_CONFIG = LLaMAConfig(
    dim=64,
    n_layers=3,
    n_heads=4,
)

LLAMA_CONFIG_DICT = {
    "7b": LLAMA_7B_CONFIG,
    "debug": DEBUG_CONFIG,
}
