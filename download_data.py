#!/usr/bin/env python3
"""
Download script for Prot2Text-V2 dataset from HuggingFace.
This script automatically downloads the required CSV files.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_huggingface_cli():
    """Check if huggingface-hub CLI is available."""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_huggingface_hub():
    """Install huggingface-hub if not available."""
    print("Installing huggingface-hub...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Failed to install huggingface-hub")
        return False

def download_with_hf_cli(data_dir: str):
    """Download data using huggingface-cli."""
    dataset_name = "habdine/Prot2Text-Data"
    
    # Files to download
    files_to_download = ["train.csv", "eval.csv", "test.csv"]
    
    for file_name in files_to_download:
        output_path = os.path.join(data_dir, file_name)
        
        if os.path.exists(output_path):
            print(f"✓ {file_name} already exists, skipping...")
            continue
        
        print(f"Downloading {file_name}...")
        try:
            cmd = [
                "huggingface-cli", "download", 
                dataset_name, file_name,
                "--local-dir", data_dir,
                "--local-dir-use-symlinks", "False"
            ]
            subprocess.run(cmd, check=True)
            print(f"✓ Downloaded {file_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {file_name}: {e}")
            if file_name == "test.csv":
                print("  Note: test.csv might not be available yet")
            else:
                return False
    
    return True

def download_with_python():
    """Download data using Python huggingface_hub library."""
    try:
        from huggingface_hub import hf_hub_download
        
        dataset_name = "habdine/Prot2Text-Data"
        data_dir = "./data"
        
        # Files to download
        files_to_download = ["train.csv", "eval.csv", "test.csv"]
        
        for file_name in files_to_download:
            output_path = os.path.join(data_dir, file_name)
            
            if os.path.exists(output_path):
                print(f"✓ {file_name} already exists, skipping...")
                continue
            
            print(f"Downloading {file_name}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=file_name,
                    local_dir=data_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✓ Downloaded {file_name}")
            except Exception as e:
                print(f"✗ Failed to download {file_name}: {e}")
                if file_name == "test.csv":
                    print("  Note: test.csv might not be available yet")
                else:
                    return False
        
        return True
    except ImportError:
        print("huggingface_hub library not found")
        return False

def manual_download_instructions():
    """Print manual download instructions."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("If automatic download fails, you can manually download the files:")
    print()
    print("1. Visit: https://huggingface.co/datasets/habdine/Prot2Text-Data")
    print("2. Download the following files:")
    print("   - train.csv")
    print("   - eval.csv")
    print("   - test.csv (if available)")
    print("3. Place them in the ./data directory")
    print()
    print("Expected directory structure:")
    print("./data/")
    print("├── train.csv")
    print("├── eval.csv")
    print("└── test.csv")
    print()

def main():
    """Main download function."""
    print("="*60)
    print("PROT2TEXT-V2 DATA DOWNLOAD")
    print("="*60)
    
    # Create data directory
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data directory: {os.path.abspath(data_dir)}")
    
    # Check if files already exist
    required_files = ["train.csv", "eval.csv"]
    existing_files = [f for f in required_files if os.path.exists(os.path.join(data_dir, f))]
    
    if len(existing_files) == len(required_files):
        print("✓ All required files already exist!")
        print("Files found:")
        for f in existing_files:
            file_path = os.path.join(data_dir, f)
            print(f"  - {f} ({os.path.getsize(file_path)} bytes)")
        return True
    
    # Try different download methods
    print("Attempting to download data...")
    
    # Method 1: HuggingFace CLI
    if check_huggingface_cli():
        print("Using huggingface-cli for download...")
        if download_with_hf_cli(data_dir):
            print("✓ Download completed successfully!")
            return True
    else:
        print("huggingface-cli not found, trying to install...")
        if install_huggingface_hub() and check_huggingface_cli():
            if download_with_hf_cli(data_dir):
                print("✓ Download completed successfully!")
                return True
    
    # Method 2: Python library
    print("Trying Python huggingface_hub library...")
    if download_with_python():
        print("✓ Download completed successfully!")
        return True
    
    # Method 3: Manual instructions
    print("✗ Automatic download failed!")
    manual_download_instructions()
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Run: python validate_data.py")
        print("2. Run: python run_qwen_training.py")
    else:
        print("\n" + "="*60)
        print("DOWNLOAD FAILED!")
        print("="*60)
        print("Please follow the manual download instructions above.")
    
    sys.exit(0 if success else 1)
