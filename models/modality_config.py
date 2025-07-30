from transformers import PretrainedConfig
class ModalityAdapterConfig(PretrainedConfig):
    """Configuration class of the 2-layer non-linear adapter."""
    model_type = "modality_adapter"  # unique identifier of the model

    def __init__(
            self, 
            input_dim: int, 
            intermediate_dim: int,
            output_dim: int, 
            dropout_rate: float = 0.3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate