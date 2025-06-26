"""
ESM-3 → ModalityAdapter → LLaMA-instruct causal LM.
"""

from typing import List, Optional, Tuple, Union

import torch
import esm
from esm.data import Alphabet

from transformers import (
    PreTrainedModel,
    Cache,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import LlamaForCausalLM

from .configuration_esm3llama_instruct import ESM3LlamaInstructConfig
from .configuration_esm3llama_instruct import ModalityAdapterConfig
from .modeling_esm2llama_instruct import ModalityAdapter


class ESM3LlamaInstructForCausalLM(PreTrainedModel):
    """
    Protein-to-text LM with ESM-3 encoder.

    - Expects `protein_input_seqs: List[str]` (raw amino-acid strings).
    - Internally tokenizes via `esm`’s Alphabet batch_converter.
    - Projects with ModalityAdapter → feeds LLaMA decoder.
    """

    config_class = ESM3LlamaInstructConfig

    def __init__(
        self,
        config: Optional[ESM3LlamaInstructConfig] = None,
        llama_decoder: Optional[LlamaForCausalLM] = None,
        **kwargs,
    ):
        super().__init__(config)

        self.esm_encoder, self.alphabet = esm.pretrained.esm3_sm_open_v1()
        config.esm_hidden_size = self.esm_encoder.embed_dim

        config.adapter_config.input_dim = config.esm_hidden_size
        self.adapter = ModalityAdapter(config.adapter_config)

        self.llama_decoder = (
            llama_decoder
            if llama_decoder is not None
            else LlamaForCausalLM(config.llama_config)
        )

        self.post_init()

    def _embed_proteins(
        self, seqs: List[str], device: torch.device
    ) -> torch.FloatTensor:
        """
        Tokenize with ESM-3’s batch_converter and return (B, L, D) reprs.
        """
        _, _, toks = self.alphabet.get_batch_converter()(
            [(f"id{i}", seq) for i, seq in enumerate(seqs)]
        )
        toks = toks.to(device)
        # ESM-3 returns (representations, logits, contacts)
        return self.esm_encoder(toks, return_contacts=False)[0]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        protein_input_seqs: Optional[List[str]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_encoder_outputs: bool = False,
        return_adapter_outputs: bool = False,
        return_decoder_inputs: bool = False,
        **decoder_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 1) Encode protein sequences
        assert protein_input_seqs is not None, "`protein_input_seqs` required"
        device = (
            input_ids.device
            if input_ids is not None
            else torch.device("cpu")
        )
        prot_repr = self._embed_proteins(protein_input_seqs, device)
        if return_encoder_outputs:
            return prot_repr

        # 2) Project into LLaMA space
        projected = self.adapter(prot_repr)
        if return_adapter_outputs:
            mask = torch.ones(
                projected.size()[:2],
                device=projected.device,
                dtype=torch.long,
            )
            return projected, mask

        # 3) Prepare decoder inputs
        inputs_embeds, new_mask = self.prepare_decoder_inputs(
            input_ids=input_ids,
            encoder_hidden_states=projected,
            attention_mask=attention_mask,
        )
        if return_decoder_inputs:
            return inputs_embeds, new_mask

        # 4) Run decoder
        return self.llama_decoder(
            input_ids=None,
            attention_mask=new_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
            **decoder_kwargs,
        )

    def generate(
        self,
        inputs: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        protein_input_seqs: Optional[List[str]] = None,
        **generate_kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Build prompt embeddings then invoke LLaMA’s generate
        prompt_embeds, prompt_mask = self(
            input_ids=inputs,
            attention_mask=attention_mask,
            protein_input_seqs=protein_input_seqs,
            return_decoder_inputs=True,
            return_dict=False,
        )
        return self.llama_decoder.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            **generate_kwargs,
        )
