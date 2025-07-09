"""
ESM Cambrian → ModalityAdapter → Qwen-14B (or any HF CausalLM) in one class.
"""

from typing import Optional, Union

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    PreTrainedModel
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_esmc_llm import ESMCLLMConfig
from .modeling_esm2llama_instruct import ModalityAdapter


class ESMCambrianLLMInstructForCausalLM(PreTrainedModel):
    """
    Protein-to-text LM with ESM Cambrian (600M) encoder.
    Simplified architecture matching Esm2LlamaInstructForCausalLM pattern.

    ESMCambrianLLMInstructForCausalLM = ESM Cambrian + ModalityAdapter + LLM
    """

    config_class = ESMCLLMConfig

    def __init__(
        self,
        config: Optional[ESMCLLMConfig] = None,
        llm_decoder: Optional[PreTrainedModel] = None,
        **kwargs,
    ):
        if config is None:
            raise ValueError("config is required for ESMCambrianLLMInstructForCausalLM")
        
        super().__init__(config)

        # ESM Cambrian encoder - use ESM++ implementation with built-in tokenizer
        self.esm_encoder = AutoModelForMaskedLM.from_pretrained(
            "Synthyra/ESMplusplus_large",  # ESM++ provides HuggingFace compatibility
            trust_remote_code=True
        )
        
        # ESM++ provides a built-in tokenizer
        self.esm_tokenizer = self.esm_encoder.tokenizer
        
        # LLM decoder (default: Qwen-14B)
        model_name = getattr(config, 'llm_model_name', 'qwen/Qwen-14B')
        self.llm_decoder = (
            llm_decoder
            if llm_decoder is not None
            else AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto"
            )
        )
        
        # Set adapter dimensions and create adapter
        config.adapter_config.input_dim = self.esm_encoder.config.hidden_size
        config.adapter_config.output_dim = self.llm_decoder.config.hidden_size
        self.adapter = ModalityAdapter(config.adapter_config)

        self.post_init()

    def prepare_decoder_inputs(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Prepare decoder inputs using placeholder replacement (same as Esm2LlamaInstruct).
        Simpler than concatenation approach.
        """
        batch_size, seq_len = input_ids.size()
        _, encoder_seq_len, _ = encoder_hidden_states.size()
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.long,
                device=input_ids.device
            )
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_seq_len),
                dtype=torch.long,
                device=encoder_hidden_states.device
            )
        
        # Get text embeddings
        inputs_embeds = self.llm_decoder.get_input_embeddings()(input_ids)
        
        # Replace placeholder tokens with protein embeddings (if placeholder_id exists)
        if hasattr(self.config, 'placeholder_id'):
            placeholder_mask = input_ids == self.config.placeholder_id
            encoder_mask = encoder_attention_mask.bool()
            inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]
        
        return inputs_embeds, attention_mask

    def forward(
        self,
        # Standard text inputs (matching esm2llama_instruct interface)
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Protein inputs (matching esm2llama_instruct interface)
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_position_ids: Optional[torch.LongTensor] = None,
        protein_head_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        # Control arguments
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_encoder_outputs: bool = False,
        return_adapter_outputs: bool = False,
        return_decoder_inputs: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        Forward pass supporting tokenized inputs only (matching esm2llama_instruct interface).
        
        Args:
            input_ids: Text token IDs (standard transformers interface)
            attention_mask: Text attention mask
            position_ids: Position indices for text tokens
            past_key_values: Cached key-value pairs for generation
            labels: Labels for training
            protein_input_ids: Tokenized protein sequences (standard interface)
            protein_attention_mask: Attention mask for protein sequences
            protein_position_ids: Position indices for protein tokens
            protein_head_mask: Head mask for protein attention
            protein_inputs_embeds: Pre-computed protein embeddings
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict output
            return_encoder_outputs: Whether to return encoder outputs only
            return_adapter_outputs: Whether to return adapter outputs only
            return_decoder_inputs: Whether to return decoder inputs only
            cache_position: Cache position for generation
            
        Returns:
            Model outputs (tuple or CausalLMOutputWithPast)
        """
        # ESM Cambrian encoder forward
        encoder_output = self.esm_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            return_dict=True
        )
        encoder_hidden_states = encoder_output.last_hidden_state
        encoder_attention_mask = protein_attention_mask
        
        if return_encoder_outputs:
            return encoder_output

        # Adapter forward
        adapter_output = self.adapter(encoder_hidden_states)
        if return_adapter_outputs:
            return adapter_output, encoder_attention_mask

        # Decoder input preparation
        inputs_embeds, attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids,
            encoder_hidden_states=adapter_output,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        if return_decoder_inputs:
            return inputs_embeds, attention_mask

        # LLM decoder forward
        return self.llm_decoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position
        )

    def generate(
        self,
        inputs: torch.LongTensor,  # alias for input_ids (standard interface)
        attention_mask: Optional[torch.LongTensor] = None,
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        protein_inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text given protein sequences and text prompts.
        Compatible with ESM2LlamaInstruct interface.
        
        Args:
            inputs: Input token IDs (standard transformers interface, alias for input_ids)
            attention_mask: Attention mask for input tokens
            protein_input_ids: Tokenized protein sequences
            protein_attention_mask: Attention mask for protein sequences
            protein_inputs_embeds: Pre-computed protein embeddings
            **kwargs: Additional generation arguments
            
        Returns:
            Generated sequences
        """
        # Get prompt embeddings using forward pass
        prompt_embeds, prompt_mask = self(
            input_ids=inputs,
            attention_mask=attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
            protein_inputs_embeds=protein_inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            return_decoder_inputs=True
        )
        
        # Do generate on llm_decoder
        return self.llm_decoder.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            **kwargs,
        )

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for all submodules that support it.
        Attention! Model needs to be in train mode before calling this method.
        """
        if hasattr(self.esm_encoder, "gradient_checkpointing_enable"):
            self.esm_encoder.gradient_checkpointing_enable()
        if hasattr(self.llm_decoder, "gradient_checkpointing_enable"):
            self.llm_decoder.gradient_checkpointing_enable()
        # Simple adapter no need to implement gradient checkpointing

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for all submodules that support it.
        """
        if hasattr(self.esm_encoder, "gradient_checkpointing_disable"):
            self.esm_encoder.gradient_checkpointing_disable()
        if hasattr(self.llm_decoder, "gradient_checkpointing_disable"):
            self.llm_decoder.gradient_checkpointing_disable()
