"""
ESM Cambrian → ModalityAdapter → LLaMA-instruct causal LM.
"""

from typing import List, Optional, Tuple, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedModel
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import LlamaForCausalLM

from .configuration_esmcllama_instruct import ESMCLlamaInstructConfig
from .configuration_esmcllama_instruct import ModalityAdapterConfig
from .modeling_esm2llama_instruct import ModalityAdapter


class ESMCambrianLlamaInstructForCausalLM(PreTrainedModel):
    """
    Protein-to-text LM with ESM Cambrian (600M) encoder.
    """

    config_class = ESMCLlamaInstructConfig

    def __init__(
        self,
        config: Optional[ESMCLlamaInstructConfig] = None,
        llama_decoder: Optional[LlamaForCausalLM] = None,
        **kwargs,
    ):
        super().__init__(config)

        # load ESM Cambrian 600M with HF transformers
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            "EvolutionaryScale/esmc-600m-2024-12",
            trust_remote_code=True
        )
        self.esm_encoder = AutoModelForMaskedLM.from_pretrained(
            "EvolutionaryScale/esmc-600m-2024-12",
            trust_remote_code=True
        )
        # embedding dimension for adapter
        config.esm_hidden_size = self.esm_encoder.config.hidden_size

        config.adapter_config.input_dim = config.esm_hidden_size
        self.adapter = ModalityAdapter(config.adapter_config)

        self.llama_decoder = (
            llama_decoder
            if llama_decoder is not None
            else LlamaForCausalLM(config.llama_config)
        )

        self.post_init()

    def _embed_proteins(
        self,
        seqs: List[str],
        device: torch.device
    ) -> torch.FloatTensor:
        """
        Tokenize with HF Cambrian tokenizer and return (B, L, D) reprs.
        """
        enc = self.esm_tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = self.esm_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state

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
