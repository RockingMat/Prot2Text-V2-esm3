from transformers import PretrainedConfig
from .modality_config import ModalityAdapterConfig


class ESMCConfig(PretrainedConfig):
    """Configuration for ESM Cambrian + LLM model."""
    model_type = "esmC_llama_instruct"
    llm_model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    esm_model_name: str = "esmc_600m"
    
    def __init__(
            self, 
            # model components
            adapter_config: ModalityAdapterConfig,
            llm_config: PretrainedConfig, 
            # standalone attributes
            placeholder_id: int = 128003
    ):
        super().__init__()
        self.adapter_config = adapter_config
        self.llm_config = llm_config
        self.placeholder_id = placeholder_id
