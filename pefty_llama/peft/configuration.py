from dataclasses import dataclass, field

PEFT_PREFIX = "prefix"
PEFT_PROMPT = "prompt"
PEFT_ADAPTER = "adapter"
PEFT_PREFIX_ADAPTER = "prefix_adapter"
PEFT_LORA = "lora"
PEFT_IA3 = "ia3"
PEFT_BITFIT = "bitfit"
NO_PEFT = "nothing"

ADAPTER_VERSION_HOULSBY = "houlsby"
ADAPTER_VERSION_PFEIFFER = "pfeiffer"


@dataclass
class PeftConfig:
    peft_mode: str = field()

    # Used by prompt, prefix, prefix_adapter
    num_prefix_tokens: int = field(default=16)

    # Prefix
    prefix_use_mlp: bool = field(default=True)
    prefix_mlp_intermediate_size: int = field(default=None)

    # LoRA
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)

    # Adapter
    adapter_hidden_size: int = field(default=64)
    adapter_version: str = field(default=ADAPTER_VERSION_PFEIFFER)  # houlsby, pfeiffer

    def check(self):
        assert self.peft_mode in (
            PEFT_PREFIX, PEFT_PREFIX_ADAPTER, PEFT_PROMPT, PEFT_ADAPTER,
            PEFT_IA3, PEFT_BITFIT,
            NO_PEFT,
        )
