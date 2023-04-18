from dataclasses import dataclass

PEFT_PREFIX = "prefix"
PEFT_PREFIX_ADAPTER = "prefix_adapter"
PEFT_PROMPT = "prefix_adapter"
PEFT_HOULSBY_ADAPTER = "houlsby_adapter"
PEFT_PFEIFFER_ADAPTER = "pfeiffer_adapter"
PEFT_LORA = "lora"
PEFT_IA3 = "ia3"
PEFT_BITFIT = "bitfit"
NO_PEFT = "nothing"


@dataclass
class PeftConfig:
    peft_mode: str

    # Used by prompt, prefix, prefix_adapter
    num_prefix_tokens: int = None

    # Prefix
    prefix_use_mlp: bool = True
    prefix_mlp_hidden_size: int = None

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16

    # IA3

    def check(self):
        assert self.peft_mode in (
            PEFT_PREFIX, PEFT_PREFIX_ADAPTER, PEFT_PROMPT, PEFT_HOULSBY_ADAPTER, PEFT_PFEIFFER_ADAPTER,
            PEFT_IA3, PEFT_BITFIT,
            NO_PEFT,
        )
