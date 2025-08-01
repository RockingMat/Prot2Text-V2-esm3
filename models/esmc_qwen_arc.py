"""
ESM C → ModalityAdapter → Qwen3-14B (or any HF CausalLM) in one class.

Pipeline:
1. Raw protein sequences → ESM C encoder → protein embeddings
2. Protein embeddings → ModalityAdapter → aligned embeddings  
3. Text with placeholder tokens → LLM embeddings
4. Replace placeholder tokens with aligned protein embeddings
5. Combined embeddings → Qwen3-14B → text output
"""

from typing import Optional, Union, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from esm.models.esmc import ESMC
from esm.utils import encoding
from esm.utils.misc import stack_variable_length_tensors

from .esmc_config import ESMCConfig
from .modeling_esm2llama_instruct import ModalityAdapter


class ESMCQwen(PreTrainedModel):
    """
    ESMCQwen = ESM C + ModalityAdapter + LLM
    """
    def __init__(
        self,
        config: ESMCConfig,
        esm_encoder: ESMC,
        adapter: ModalityAdapter,
        llm_decoder: AutoModelForCausalLM,
        llm_tokenizer: PreTrainedTokenizer
    ):
        super().__init__(config)
        self.esm_encoder = esm_encoder
        self.adapter = adapter
        self.llm_decoder = llm_decoder
        self.llm_tokenizer = llm_tokenizer

    def encode_protein_sequences(self, protein_sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein sequences using ESM C

        1. Tokenize sequences individually using the tokenizer
        2. Batch with padding using stack_variable_length_tensors 
        3. Create ESMProteinTensor with batched token IDs
        4. Get embeddings from the tensor (not individual sequences)
        
        Args:
            protein_sequences: List of protein sequences (strings)
            
        Returns:
            Tuple of (protein_embeddings, attention_mask):
            - protein_embeddings: tensor (batch_size, seq_len, hidden_size)
            - attention_mask: tensor (batch_size, seq_len) with 1 for valid positions, 0 for padded
        """
        pad_token_id = self.esm_encoder.tokenizer.pad_token_id
        assert pad_token_id is not None
        
        # This uses the same approach as ESMC's _tokenize method
        tokenized_sequences = [
            encoding.tokenize_sequence(seq, self.esm_encoder.tokenizer, add_special_tokens=True)
            for seq in protein_sequences
        ]
        
        batched_token_ids = stack_variable_length_tensors(
            tokenized_sequences,
            constant_value=pad_token_id,
        ).to(next(self.esm_encoder.parameters()).device)
        
        output = self.esm_encoder.forward(sequence_tokens=batched_token_ids)

        attention_mask = (batched_token_ids != pad_token_id).long()
        
        encoder_hidden_states = output.embeddings
        adapter_dtype = next(self.adapter.parameters()).dtype
        encoder_hidden_states = encoder_hidden_states.to(adapter_dtype)
        
        return encoder_hidden_states, attention_mask

    def prepare_decoder_inputs(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Embed and replace placeholder in `input_ids` by encoder hidden states.
        
        Args:
            input_ids: Token IDs containing placeholder tokens
            encoder_hidden_states: Protein embeddings from adapter
            attention_mask: Attention mask for input tokens
            encoder_attention_mask: Attention mask for encoder outputs
            
        Returns:
            Tuple of (inputs_embeds, attention_mask) ready for LLM
        """
        # preparation
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
        
        # Get text embeddings from input_ids
        inputs_embeds = self.llm_decoder.get_input_embeddings()(input_ids)
        
        # Placeholder replacement: replace placeholder tokens with protein embeddings
        placeholder_mask = input_ids == self.config.placeholder_id
        encoder_mask = encoder_attention_mask.bool()
        
        # Verify that we have the right number of placeholder tokens
        num_placeholders = placeholder_mask.sum(dim=1)
        num_protein_tokens = encoder_mask.sum(dim=1)
        
        if not torch.all(num_placeholders == num_protein_tokens):
            raise ValueError(
                f"Number of placeholder tokens ({num_placeholders.tolist()}) "
                f"must match number of protein tokens ({num_protein_tokens.tolist()})"
            )
        
        # Replace placeholder embeddings with protein embeddings
        inputs_embeds[placeholder_mask] = encoder_hidden_states[encoder_mask]
        
        return inputs_embeds, attention_mask

    def forward(
        self,
        # Protein inputs (required)
        protein_sequences: List[str],
        # Text inputs (optional for encoder-only mode)
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Special flags
        return_encoder_outputs: bool = False,
        return_decoder_inputs: bool = False,
        # Generation parameters
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        Simple forward pass: protein sequences → ESM C → adapter → Qwen3-14B
        
        Args:
            protein_sequences: Raw protein sequences (required)
            input_ids: Text token IDs with placeholder tokens (optional for encoder-only mode)
            attention_mask: Text attention mask
            labels: Labels for training
            return_encoder_outputs: If True, return only encoder outputs (for contrastive learning)
            return_decoder_inputs: If True, return prepared decoder inputs
            
        Returns:
            Model outputs or encoder outputs if return_encoder_outputs=True
        """
        # Step 1: Encode protein sequences
        protein_embeddings, protein_attention_mask = self.encode_protein_sequences(protein_sequences)
        
        # Step 2: Adapt protein embeddings
        adapted_embeddings = self.adapter(protein_embeddings)
        
        # If only encoder outputs are requested (for contrastive learning)
        if return_encoder_outputs:
            return (adapted_embeddings,)
        
        # Step 3: Full forward pass with LLM using placeholder replacement
        if input_ids is None:
            raise ValueError("input_ids must be provided for full forward pass")
            
        # Use the protein attention mask returned by encode_protein_sequences
        # (no need to create a new one since we now have proper masking for padded positions)
        
        # Prepare decoder inputs with placeholder replacement
        inputs_embeds, final_attention_mask = self.prepare_decoder_inputs(
            input_ids=input_ids,
            encoder_hidden_states=adapted_embeddings,
            attention_mask=attention_mask,
            encoder_attention_mask=protein_attention_mask,
        )
        
        # If only decoder inputs are requested (for generation setup)
        if return_decoder_inputs:
            return inputs_embeds, final_attention_mask
        
        # Step 4: Forward through LLM
        return self.llm_decoder(
            input_ids=None,
            attention_mask=final_attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        protein_sequences: List[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text given protein sequences and text prompts using placeholder replacement.
        
        Args:
            input_ids: Input token IDs (text prompt with placeholder tokens)
            attention_mask: Attention mask for input tokens
            protein_sequences: List of protein sequences (strings)
            **kwargs: Additional generation arguments
            
        Returns:
            Generated sequences
        """
        if protein_sequences is None:
            raise ValueError("protein_sequences must be provided for generation")
        
        prompt_inputs_embeds, prompt_attention_mask = self(
            protein_sequences=protein_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
            return_decoder_inputs=True
        )
        
        return self.llm_decoder.generate(
            inputs_embeds=prompt_inputs_embeds,
            attention_mask=prompt_attention_mask,
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
