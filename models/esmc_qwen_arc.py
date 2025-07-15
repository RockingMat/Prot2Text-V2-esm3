"""
ESM C → ModalityAdapter → Qwen3-14B (or any HF CausalLM) in one class.

Simplified pipeline:
1. Raw protein sequences → ESM C encoder → protein embeddings
2. Protein embeddings → ModalityAdapter → aligned embeddings  
3. Aligned embeddings + text tokens → Qwen3-14B → text output
"""

from typing import Optional, Union, List

import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from .esmc_config import ESMCConfig
from .modeling_esm2llama_instruct import ModalityAdapter


class ESMCQwen(PreTrainedModel):
    """
    ESMCQwen = ESM C + ModalityAdapter + LLM
    """
    def __init__(
        self,
        esm_encoder: ESMC,
        adapter: ModalityAdapter,
        llm_decoder: AutoModelForCausalLM
    ):
        config = ESMCConfig(
            esm_config=esm_encoder.config,
            adapter_config=adapter.config,
            llm_config=llm_decoder.config
        ) 
        super().__init__(config)
        self.esm_encoder = esm_encoder
        self.adapter = adapter
        self.llm_decoder = llm_decoder

    def encode_protein_sequences(self, protein_sequences: List[str]) -> torch.Tensor:
        """
        Encode protein sequences using ESM C.
        
        Args:
            protein_sequences: List of protein sequences (strings)
            
        Returns:
            Protein embeddings tensor (batch_size, seq_len, hidden_size)
        """
        # Convert to ESMProtein objects
        proteins = [ESMProtein(sequence=seq) for seq in protein_sequences]
        
        # Encode each protein
        protein_tensors = [self.esm_encoder.encode(protein) for protein in proteins]
        
        # Get embeddings using logits API
        batch_embeddings = []
        for protein_tensor in protein_tensors:
            logits_output = self.esm_encoder.logits(
                protein_tensor, 
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            batch_embeddings.append(logits_output.embeddings)
        
        # Stack embeddings
        encoder_hidden_states = torch.stack(batch_embeddings, dim=0)
        
        return encoder_hidden_states

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
        # Text inputs
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Protein inputs (simplified - use raw sequences)
        protein_sequences: Optional[List[str]] = None,
        # Legacy support for tokenized proteins
        protein_input_ids: Optional[torch.LongTensor] = None,
        protein_attention_mask: Optional[torch.LongTensor] = None,
        # Generation parameters
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        Simple forward pass: protein sequences → ESM C → adapter → Qwen3-14B
        
        Args:
            input_ids: Text token IDs
            attention_mask: Text attention mask
            labels: Labels for training
            protein_sequences: Raw protein sequences (preferred)
            protein_input_ids: Tokenized protein sequences (legacy)
            protein_attention_mask: Protein attention mask (legacy)
            
        Returns:
            Model outputs
        """
        # Step 1: Encode protein sequences
        if protein_sequences is not None:
            # Use raw sequences (preferred)
            protein_embeddings = self.encode_protein_sequences(protein_sequences)
        elif protein_input_ids is not None:
            # Legacy support - convert tokenized sequences back to strings
            # This is a fallback for existing training scripts
            raise NotImplementedError("Legacy protein_input_ids support not implemented. Use protein_sequences instead.")
        else:
            # Create dummy embeddings for testing
            batch_size = input_ids.shape[0]
            seq_len = 512
            protein_embeddings = torch.zeros(
                (batch_size, seq_len, 1280),
                dtype=torch.float32,
                device=input_ids.device
            )
        
        # Step 2: Adapt protein embeddings
        adapted_embeddings = self.adapter(protein_embeddings)
        
        # Step 3: Prepare inputs for LLM
        # Get text embeddings
        text_embeddings = self.llm_decoder.get_input_embeddings()(input_ids)
        
        # For now, use simple concatenation (can be improved with placeholder replacement)
        # Concatenate protein embeddings with text embeddings
        combined_embeddings = torch.cat([adapted_embeddings, text_embeddings], dim=1)
        
        # Create combined attention mask
        protein_mask = torch.ones(
            (adapted_embeddings.shape[0], adapted_embeddings.shape[1]),
            dtype=torch.long,
            device=input_ids.device
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        combined_attention_mask = torch.cat([protein_mask, attention_mask], dim=1)
        
        # Adjust labels if provided
        if labels is not None:
            # Add -100 labels for protein embeddings (don't compute loss on protein part)
            protein_labels = torch.full(
                (labels.shape[0], adapted_embeddings.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            combined_labels = torch.cat([protein_labels, labels], dim=1)
        else:
            combined_labels = None
        
        # Step 4: Forward through LLM
        return self.llm_decoder(
            input_ids=None,
            attention_mask=combined_attention_mask,
            inputs_embeds=combined_embeddings,
            labels=combined_labels,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        protein_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text given protein sequences and text prompts.
        
        Args:
            input_ids: Input token IDs (text prompt)
            attention_mask: Attention mask for input tokens
            protein_sequences: List of protein sequences (strings)
            **kwargs: Additional generation arguments
            
        Returns:
            Generated sequences
        """
        # Get combined embeddings using forward pass
        # We'll use forward pass to get the combined embeddings
        with torch.no_grad():
            # Step 1: Encode protein sequences
            if protein_sequences is not None:
                protein_embeddings = self.encode_protein_sequences(protein_sequences)
            else:
                # Create dummy embeddings
                batch_size = input_ids.shape[0]
                seq_len = 512
                protein_embeddings = torch.zeros(
                    (batch_size, seq_len, 1280),
                    dtype=torch.float32,
                    device=input_ids.device
                )
            
            # Step 2: Adapt protein embeddings
            adapted_embeddings = self.adapter(protein_embeddings)
            
            # Step 3: Prepare inputs for LLM
            text_embeddings = self.llm_decoder.get_input_embeddings()(input_ids)
            
            # Concatenate protein embeddings with text embeddings
            combined_embeddings = torch.cat([adapted_embeddings, text_embeddings], dim=1)
            
            # Create combined attention mask
            protein_mask = torch.ones(
                (adapted_embeddings.shape[0], adapted_embeddings.shape[1]),
                dtype=torch.long,
                device=input_ids.device
            )
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            combined_attention_mask = torch.cat([protein_mask, attention_mask], dim=1)
        
        # Step 4: Generate using LLM
        return self.llm_decoder.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
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
