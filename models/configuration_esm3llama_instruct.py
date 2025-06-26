from dataclasses import dataclass

from .configuration_esm2llama_instruct import (
    Esm2LlamaInstructConfig,
)


@dataclass
class ESM3LlamaInstructConfig(Esm2LlamaInstructConfig):
    model_type = "esm3_llama_instruct"