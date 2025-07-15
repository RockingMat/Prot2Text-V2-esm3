# Prot2Text-V2: Simple Protein Function Prediction

**Simple pipeline**: Protein sequences → ESM Cambrian → Adapter → Qwen3-14B → Function descriptions

## Quick Start

### 1. Install Dependencies

```bash
conda create -n prot2text python=3.10
conda activate prot2text

# Core requirements
pip install torch transformers>=4.51.0 peft
pip install pandas numpy biopython esm
pip install accelerate deepspeed tensorboard
```

### 2. Get Data

```bash
# Download data automatically
python download_data.py

# Validate data format
python validate_data.py
```

Data format (CSV files):
- `sequence`: Protein amino acid sequence
- `Full Name`: Protein name
- `taxon`: Organism name  
- `function`: Function description

### 3. Train Model

```bash
# Simple training (recommended)
python run_qwen_training.py

# Or with custom parameters
python scripts/train_instruct.py \
    --esm_model_name "esmc_600m" \
    --llama_path "Qwen/Qwen3-14B" \
    --root_csv_dir "./data" \
    --save_checkpoint_dir "./checkpoints" \
    --batch_size_per_device 2 \
    --num_epochs 3 \
    --lora_rank 16
```

### 4. Generate Predictions

```bash
python scripts/generate_instruct.py \
    --load_adapter_checkpoint_dir "./checkpoints" \
    --esm_model_name "esmc_600m" \
    --llama_path "Qwen/Qwen3-14B"
```

## Model Architecture

**Simple 3-step pipeline**:

1. **ESM Cambrian** (`esmc_300m` or `esmc_600m`) → protein embeddings
2. **Modality Adapter** → aligned embeddings
3. **Qwen3-14B** → function descriptions

**Key Features**:
- **No tokenization needed** for protein sequences
- **CSV-only data** (no PDB files)
- **LoRA fine-tuning** for efficiency
- **Thinking capabilities** from Qwen3-14B

## Data Pipeline

```
Raw protein sequences → ESM Cambrian → Adapter → Qwen3-14B → Text output
```

**Benefits**:
- No complex graph processing
- No PDB file requirements
- Faster training and inference
- Simpler debugging

## Training Parameters

**Essential parameters**:
- `--esm_model_name`: `esmc_300m` or `esmc_600m`
- `--llama_path`: `Qwen/Qwen3-14B`
- `--batch_size_per_device`: 1-4 (depends on GPU memory)
- `--lora_rank`: 8-16 (LoRA rank)
- `--num_epochs`: 3-5

**Debug mode**:
```bash
python run_qwen_training.py debug  # Test with limited data
```

## Requirements

- **Python**: 3.10+ (required for ESM package)
- **GPU**: NVIDIA GPU with 16GB+ memory
- **Dependencies**: transformers>=4.51.0, torch, peft, esm
- **Data**: CSV files with protein sequences and descriptions

## Troubleshooting

1. **GPU memory issues**: Reduce `batch_size_per_device` to 1
2. **ESM import errors**: Ensure Python 3.10+ and `pip install esm`
3. **Qwen3 errors**: Use `trust_remote_code=True` and `transformers>=4.51.0`
4. **Training too slow**: Use `debug` mode first

## API Usage

```python
from models.modeling_esmc_llm import ESMCambrianLLMInstructForCausalLM
from models.configuration_esmc_llm import ESMCLLMConfig

# Create model
config = ESMCLLMConfig(llm_model_name="Qwen/Qwen3-14B")
model = ESMCambrianLLMInstructForCausalLM(config=config)

# Forward pass
protein_sequences = ["MKLLMVGSG", "MTTLVQAVS"]
input_ids = tokenizer("Describe this protein:", return_tensors="pt")["input_ids"]

outputs = model(
    input_ids=input_ids,
    protein_sequences=protein_sequences
)
```

## File Structure

```
├── README.md                 # This file
├── run_qwen_training.py      # Simple training script
├── scripts/
│   ├── train_instruct.py     # Main training script
│   └── generate_instruct.py  # Generation script
├── models/
│   ├── modeling_esmc_llm.py  # Main model class
│   └── configuration_esmc_llm.py  # Model config
├── dataset/
│   └── dataloader_light.py   # Dataset loading
└── data/
    ├── train.csv             # Training data
    ├── eval.csv              # Validation data
    └── test.csv              # Test data (optional)
```

## Citation

```bibtex
@article{prot2text_v2,
  title={Prot2Text-V2: Simple Protein Function Prediction},
  author={[Authors]},
  year={2024}
}
```
