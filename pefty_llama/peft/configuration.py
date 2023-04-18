from dataclasses import dataclass

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
    peft_mode: str

    # Used by prompt, prefix, prefix_adapter
    num_prefix_tokens: int = 16

    # Prefix
    prefix_use_mlp: bool = True
    prefix_mlp_intermediate_size: int = None

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16

    # Adapter
    adapter_hidden_size: int = 64
    adapter_version: str = ADAPTER_VERSION_PFEIFFER  # houlsby, pfeiffer

    def check(self):
        assert self.peft_mode in (
            PEFT_PREFIX, PEFT_PREFIX_ADAPTER, PEFT_PROMPT, PEFT_ADAPTER,
            PEFT_IA3, PEFT_BITFIT,
            NO_PEFT,
        )
