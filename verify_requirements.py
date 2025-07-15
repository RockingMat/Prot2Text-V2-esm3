#!/usr/bin/env python3
"""
Package Requirements Verification Script
This script verifies that all required packages are installed and compatible.
"""

import sys
import subprocess
import importlib

def check_package(package_name, min_version=None):
    """Check if a package is installed and optionally verify version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version:
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"{package_name} {version} < {min_version}"
        
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} not installed"

def main():
    print("Prot2Text-V2 Package Requirements Verification")
    print("=" * 50)
    
    # Core requirements for CSV-only workflow
    requirements = [
        ("torch", None),
        ("transformers", "4.51.0"),
        ("tokenizers", None),
        ("accelerate", None),
        ("peft", None),
        ("pandas", None),
        ("numpy", None),
        ("tqdm", None),
        ("deepspeed", None),
        ("tensorboard", None),
        ("evaluate", None),
        ("nltk", None),
        ("rouge_score", None),
        ("jiwer", None),
        ("biopython", None),
        ("huggingface_hub", None),
        ("requests", None),
        ("chardet", None),
        ("charset_normalizer", None),
        ("multiprocess", None),
        ("sentencepiece", None),
        ("esm", None),  # ESM package for protein encoder
    ]
    
    print(f"Python version: {sys.version}")
    print()
    
    all_good = True
    
    for package, min_version in requirements:
        success, message = check_package(package, min_version)
        status = "✅" if success else "❌"
        print(f"{status} {message}")
        
        if not success:
            all_good = False
    
    print()
    
    if all_good:
        print("✅ All required packages are installed and compatible!")
    else:
        print("❌ Some packages are missing or incompatible.")
        print("Please install missing packages using:")
        print("pip install torch transformers>=4.51.0 tokenizers accelerate peft pandas numpy tqdm deepspeed tensorboard evaluate nltk rouge_score jiwer biopython huggingface-hub requests chardet charset-normalizer multiprocess sentencepiece esm")
    
    # Test critical imports for CSV-only workflow
    print("\nTesting critical imports for CSV-only workflow:")
    print("-" * 50)
    
    critical_imports = [
        "from transformers import AutoTokenizer, AutoModelForCausalLM",
        "from peft import get_peft_model, LoraConfig",
        "import torch",
        "import pandas as pd",
        "import numpy as np",
        "from esm.models.esmc import ESMC",
        "from esm.sdk.api import ESMProtein, LogitsConfig",
        "from dataset import Prot2TextLightDataset, Prot2TextLightCollater",
        "from models.modeling_esmc_llm import ESMCambrianLLMInstructForCausalLM",
        "from models.configuration_esmc_llm import ESMCLLMConfig",
    ]
    
    for import_stmt in critical_imports:
        try:
            exec(import_stmt)
            print(f"✅ {import_stmt}")
        except Exception as e:
            print(f"❌ {import_stmt} - Error: {e}")
            all_good = False
    
    print()
    
    if all_good:
        print("🎉 All systems ready for Prot2Text-V2 with Qwen3-14B!")
    else:
        print("⚠️  Some imports failed. Please check your installation.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
