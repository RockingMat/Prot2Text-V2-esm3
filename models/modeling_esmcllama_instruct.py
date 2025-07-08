"""
ESM Cambrian → ModalityAdapter → Qwen-14B (or any HF CausalLM) in one class with built-in tokenization.
"""

from typing import List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    PreTrainedModel
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_esmcllama_instruct import ESMCLLMConfig
from .modeling_esm2llama_instruct import ModalityAdapter


class ESMCambrianLLMInstructForCausalLM(PreTrainedModel):
    """
    Protein-to-text LM with ESM Cambrian (600M) encoder and built-in LLM tokenization/generation.

    Inputs:
      - `protein_input_seqs`: List[str] of amino-acid sequences
      - `text_prompts`: Optional[List[str]] of raw text prompts

    Forward:
      - Automatically tokenizes `text_prompts` if provided
      - Runs Cambrian encoder → adapter → CausalLM decoder
    Generate:
      - Accepts raw `text_prompts` and handles tokenization internally
    """

    config_class = ESMCLLMConfig

    def __init__(
        self,
        config: Optional[ESMCLLMConfig] = None,
        llm_decoder: Optional[PreTrainedModel] = None,
        **kwargs,
    ):
        super().__init__(config)

        # ESM Cambrian encoder
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            "EvolutionaryScale/esmc-600m-2024-12",
            trust_remote_code=True
        )
        self.esm_encoder = AutoModelForMaskedLM.from_pretrained(
            "EvolutionaryScale/esmc-600m-2024-12",
            trust_remote_code=True
        )
        config.esm_hidden_size = self.esm_encoder.config.hidden_size

        # Adapter between protein and LLM
        config.adapter_config.input_dim = config.esm_hidden_size
        self.adapter = ModalityAdapter(config.adapter_config)

        # LLM decoder & tokenizer (default: Qwen-14B)
        model_name = getattr(config, 'llm_model_name', 'qwen/Qwen-14B')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.llm_decoder = (
            llm_decoder
            if llm_decoder is not None
            else AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto"
            )
        )

        self.post_init()

    def _embed_proteins(
        self,
        seqs: List[str],
        device: torch.device
    ) -> torch.FloatTensor:
        enc = self.esm_tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )
        toks = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        outputs = self.esm_encoder(
            input_ids=toks,
            attention_mask=mask,
            return_dict=True
        )
        return outputs.last_hidden_state

    def forward(
        self,
        protein_input_seqs: List[str],
        text_prompts: Optional[List[str]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_encoder_outputs: bool = False,
        return_adapter_outputs: bool = False,
        return_decoder_inputs: bool = False,
        **decoder_kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        # 1) Tokenize prompts if raw text provided
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if text_prompts is not None:
            tok = self.llm_tokenizer(
                text_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)

        # 2) Encode protein sequences
        prot_repr = self._embed_proteins(protein_input_seqs, device)
        if return_encoder_outputs:
            return prot_repr

        # 3) Project into LLM space
        projected = self.adapter(prot_repr)
        if return_adapter_outputs:
            mask = torch.ones(
                projected.size()[:2],
                device=projected.device,
                dtype=torch.long,
            )
            return projected, mask

        # 4) Prepare decoder inputs
        inputs_embeds, new_mask = self.prepare_decoder_inputs(
            input_ids=input_ids,
            encoder_hidden_states=projected,
            attention_mask=attention_mask,
        )
        if return_decoder_inputs:
            return inputs_embeds, new_mask

        # 5) Run decoder
        return self.llm_decoder(
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
        protein_input_seqs: List[str],
        text_prompts: Optional[List[str]] = None,
        max_length: int = 128,
        **generate_kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # handle text tokenization internally
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if text_prompts is not None:
            tok = self.llm_tokenizer(
                text_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
        else:
            raise ValueError("`text_prompts` required for generate")

        # reuse forward to get prompt embeddings
        prompt_embeds, prompt_mask = self(
            protein_input_seqs=protein_input_seqs,
            text_prompts=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_decoder_inputs=True,
            return_dict=False
        )
        return self.llm_decoder.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            max_length=max_length,
            **generate_kwargs,
        )
