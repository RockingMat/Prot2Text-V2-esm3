#!/usr/bin/env python3
"""
Data preparation script for Prot2Text-V2 training pipelines.
Run this script to prepare the dataset for contrastive learning.
Note: This version processes only CSV files (no PDB files or graph processing).
"""

import os
from transformers import AutoTokenizer
from dataset import Prot2TextLightDataset

# Configuration
SPLITS = ["train", "eval"]  # Add "test" if you have it
CSV_DIR = "./data"  # Directory containing your CSV files
DATA_ROOT_DIR = "/tmp/Prot2Text-Data"  # Local directory for processed data
LLAMA_DIR = "Qwen/Qwen3-14B"  # Updated for Qwen3-14B
ESM_MODEL_NAME = "esmc_600m"  # ESM C model

def prepare_dataset():
    """Prepare dataset for contrastive learning pipeline (CSV only)."""
    
    # Create root directory
    os.makedirs(DATA_ROOT_DIR, exist_ok=True)
    
    # Load tokenizers
    print("Loading tokenizers...")
    # For ESM C model, no tokenizer is needed as it uses raw protein sequences
    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_DIR, 
        pad_token='<|im_end|>',  # Use <|im_end|> for Qwen3 fine-tuning
        trust_remote_code=True
    )
    
    # Process each split
    for split in SPLITS:
        print(f"\nProcessing {split} split...")
        
        csv_path = os.path.join(CSV_DIR, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping {split} split.")
            continue
            
        # Use light dataset (CSV only, no PDB files)
        split_dataset = Prot2TextLightDataset(
            csv_path=csv_path,
            sequence_tokenizer=None,  # No tokenizer needed for ESM C
            description_tokenizer=llama_tokenizer,
        )
        
        print(f"Completed processing {split} split.")
        print(f"Dataset size: {len(split_dataset)}")

if __name__ == "__main__":
    # Check if CSV files exist
    required_files = [f"{split}.csv" for split in SPLITS]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(CSV_DIR, f))]
    
    if missing_files:
        print("ERROR: Missing CSV files:")
        for f in missing_files:
            print(f"  - {os.path.join(CSV_DIR, f)}")
        print("\nPlease download CSV files from: https://huggingface.co/datasets/habdine/Prot2Text-Data")
        print("And place them in the ./data directory")
        exit(1)
    
    prepare_dataset()
    print("\nData preparation complete!")
    print(f"Processed data saved to: {DATA_ROOT_DIR}")
    print("Note: This version processes only CSV files (no PDB files or graph processing).")
