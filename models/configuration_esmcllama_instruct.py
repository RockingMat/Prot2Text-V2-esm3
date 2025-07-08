from dataclasses import dataclass

from .configuration_esm2llama_instruct import (
    Esm2LlamaInstructConfig,
)


@dataclass
class ESMCLlamaInstructConfig(Esm2LlamaInstructConfig):
    model_type = "esmC_llama_instruct"