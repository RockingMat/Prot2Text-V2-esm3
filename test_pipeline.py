#!/usr/bin/env python3
"""
Test script for the simplified ESM Cambrian pipeline.
"""

import torch
from transformers import AutoTokenizer

# Test imports
try:
    from models.configuration_esmc_llm import ESMCLLMConfig, ModalityAdapterConfig
    from models.modeling_esmc_llm import ESMCambrianLLMInstructForCausalLM
    print("✓ Model imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

def test_pipeline():
    """Test the simplified pipeline with dummy data."""
    
    # Create config
    config = ESMCLLMConfig(
        llm_model_name="microsoft/DialoGPT-medium",  # Use smaller model for testing
        adapter_config=ModalityAdapterConfig(
            input_dim=1280,  # ESM C hidden size
            intermediate_dim=1024,
            output_dim=1024,  # DialoGPT hidden size
        )
    )
    
    print("✓ Configuration created")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-medium",
        pad_token='<|endoftext|>',
        trust_remote_code=True
    )
    
    print("✓ Tokenizer created")
    
    # Create model (this will be slow as it loads ESM C and the LLM)
    try:
        model = ESMCambrianLLMInstructForCausalLM(
            config=config,
            esm_model_name="esmc_300m"  # Use smaller ESM model for testing
        )
        print("✓ Model created")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        print("This is expected if ESM C is not properly installed")
        return
    
    # Test data
    protein_sequences = [
        "MKLLMVGSG",
        "MTTLVQAVS"
    ]
    
    text_prompt = "Describe this protein:"
    input_ids = tokenizer(
        [text_prompt, text_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True
    )["input_ids"]
    
    print("✓ Test data prepared")
    
    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                protein_sequences=protein_sequences,
                use_cache=False,
                return_dict=False
            )
        print(f"✓ Forward pass successful, output shape: {outputs[0].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_pipeline()
