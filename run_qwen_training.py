#!/usr/bin/env python3
"""
Updated training script for Qwen3-14B with proper tokenizer configuration.
This script addresses the specific requirements for Qwen3 models.
"""

import os
import subprocess
import sys

def setup_qwen_tokenizer():
    """Setup Qwen3-14B tokenizer with proper pad token configuration."""
    
    tokenizer_setup = """
# Qwen3-14B Tokenizer Configuration Notes:
# - Default pad_token: "<|endoftext|>"
# - For fine-tuning: Use "<|im_end|>" to avoid infinite generation
# - Available special tokens: <|im_start|>, <|im_end|>, <|endoftext|>
# - Requires trust_remote_code=True for Qwen3 models
# - Requires transformers>=4.51.0 for Qwen3 support
"""
    
    print(tokenizer_setup)
    
    # Check if we have proper tokenizer setup
    try:
        from transformers import AutoTokenizer
        
        # Test Qwen3-14B tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
        
        # Check current pad token
        print(f"Current pad token: {tokenizer.pad_token}")
        print(f"Current pad token ID: {tokenizer.pad_token_id}")
        
        # Available special tokens
        print(f"Special tokens: {tokenizer.special_tokens_map}")
        
        return True
    except Exception as e:
        print(f"Error setting up tokenizer: {e}")
        print("Make sure you have transformers>=4.51.0 installed")
        return False

def run_qwen_training(mode="instruct_only"):
    """Run training with Qwen3-14B specific configurations."""
    
    if not setup_qwen_tokenizer():
        print("Failed to setup Qwen tokenizer. Please check your installation.")
        return False
    
    # Base command for instruction tuning
    base_cmd = [
        "python", "scripts/train_instruct.py",
        "--llama_path", "Qwen/Qwen3-14B",
        "--root_csv_dir", "./data",
        "--save_checkpoint_dir", "./checkpoints/qwen3_instruct",
        "--torch_dtype", "float16",
        "--batch_size_per_device", "2",
        "--num_epochs", "3",
        "--save_every_epochs", "1",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "1e-4",
        "--scheduler_gamma", "0.95",
        "--random_seed", "42",
        "--lora_rank", "16",
        "--adapter_intermediate_dim", "2048",
        "--fix_modality_adapter", "false",
        "--include_text_fields", "true",
        "--name_dropout", "0.1",
        "--taxonomy_dropout", "0.8",
        "--train_split", "train",
        "--eval_split", "eval"
    ]
    
    # Add debug flags for testing
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        base_cmd.extend([
            "--debug_trim_train_split", "10",
            "--debug_trim_eval_split", "5",
            "--num_epochs", "1"
        ])
        print("Running in DEBUG mode with limited data...")
    
    print("Running command:")
    print(" ".join(base_cmd))
    
    # Run the training
    try:
        result = subprocess.run(base_cmd, check=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False

def main():
    """Main function."""
    
    print("="*60)
    print("QWEN3-14B TRAINING SCRIPT")
    print("="*60)
    
    # Check if data exists
    if not os.path.exists("./data/train.csv"):
        print("ERROR: ./data/train.csv not found!")
        print("Please download CSV files from: https://huggingface.co/datasets/habdine/Prot2Text-Data")
        return False
    
    if not os.path.exists("./data/eval.csv"):
        print("ERROR: ./data/eval.csv not found!")
        print("Please download CSV files from: https://huggingface.co/datasets/habdine/Prot2Text-Data")
        return False
    
    # Run training
    success = run_qwen_training()
    
    if success:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check ./checkpoints/qwen3_instruct/ for saved models")
    else:
        print("\n" + "="*60)
        print("TRAINING FAILED!")
        print("="*60)
        print("Check the error messages above for troubleshooting")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
