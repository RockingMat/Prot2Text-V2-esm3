from dataclasses import dataclass
from typing import Optional

from .configuration_esm2llama_instruct import (
    Esm2LlamaInstructConfig,
    ModalityAdapterConfig,
)


@dataclass
class ESMCLLMConfig(Esm2LlamaInstructConfig):
    """Configuration for ESM Cambrian + LLM model."""
    model_type = "esmC_llama_instruct"
    llm_model_name: str = "Qwen/Qwen3-14B"
    esm_model_name: str = "esmc_600m"
    esm_hidden_size: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.adapter_config is None:
            self.adapter_config = ModalityAdapterConfig(
                input_dim=0,
                intermediate_dim=2048,
                output_dim=0,
            )