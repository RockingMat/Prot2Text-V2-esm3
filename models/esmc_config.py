from transformers import PretrainedConfig
from .modality_config import ModalityAdapterConfig


class ESMCConfig(PretrainedConfig):
    """Configuration for ESM Cambrian + LLM model."""
    model_type = "esmC_llama_instruct"
    llm_model_name: str = "Qwen/Qwen3-14B"
    esm_model_name: str = "esmc_600m"
    
    def __init__(
            self, 
            adapter_config: ModalityAdapterConfig,
            llm_config: PretrainedConfig, 
    ):
        super().__init__()
        self.adapter_config = adapter_config
        self.llm_config = llm_config
