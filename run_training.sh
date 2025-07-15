#!/bin/bash
# Training script for Prot2Text-V2 with Qwen3-14B and ESM Cambrian

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Model paths
ESM_PATH="Synthyra/ESMplusplus_large"
LLAMA_PATH="Qwen/Qwen3-14B"

# Data directories
CSV_DIR="./data"
DATA_DIR="/tmp/Prot2Text-Data"
CHECKPOINT_DIR="./checkpoints"

# Training parameters
BATCH_SIZE=4
EPOCHS_CONTRAST=3
EPOCHS_INSTRUCT=2

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File not found: $1${NC}"
        return 1
    fi
    return 0
}

check_directory() {
    if [ ! -d "$1" ]; then
        echo -e "${YELLOW}Creating directory: $1${NC}"
        mkdir -p "$1"
    fi
}

# ============================================================================
# STEP 1: Data Preparation
# ============================================================================

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}PROT2TEXT-V2 TRAINING PIPELINE${NC}"
echo -e "${GREEN}===========================================${NC}"

echo -e "${YELLOW}Step 1: Checking data...${NC}"

# Check if CSV files exist
if ! check_file "$CSV_DIR/train.csv" || ! check_file "$CSV_DIR/eval.csv"; then
    echo -e "${RED}Missing CSV files!${NC}"
    echo "Please run: python download_data.py"
    echo "Or download manually from: https://huggingface.co/datasets/habdine/Prot2Text-Data"
    exit 1
fi

# Validate data
echo "Validating data format..."
python validate_data.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Data validation failed!${NC}"
    exit 1
fi

# Create checkpoint directory
check_directory "$CHECKPOINT_DIR"

# ============================================================================
# STEP 2: Contrastive Learning (Stage 1)
# ============================================================================

echo "Step 2: Running contrastive learning..."

python scripts/train_contrast.py \
    --llama_path "Qwen/Qwen3-14B" \
    --root_dataset_dir "/tmp/Prot2Text-Data" \
    --root_csv_dir "./data" \
    --save_checkpoint_dir "./checkpoints/contrastive" \
    --torch_dtype "float16" \
    --batch_size_per_device 4 \
    --num_epochs 10 \
    --save_every_epochs 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --scheduler_gamma 0.95 \
    --random_seed 42 \
    --contrastive_num_segments 10 \
    --train_split "train" \
    --eval_split "eval"

# ============================================================================
# STEP 3: Instruction Tuning (Stage 2)
# ============================================================================

echo "Step 3: Running instruction tuning..."

python scripts/train_instruct.py \
    --llama_path "Qwen/Qwen3-14B" \
    --root_csv_dir "./data" \
    --save_checkpoint_dir "./checkpoints/instruct" \
    --load_model_checkpoint_path "./checkpoints/contrastive/checkpoints_XXXXXX/model_checkpoint_10.pt" \
    --torch_dtype "float16" \
    --batch_size_per_device 4 \
    --num_epochs 5 \
    --save_every_epochs 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --scheduler_gamma 0.95 \
    --random_seed 42 \
    --lora_rank 16 \
    --adapter_intermediate_dim 2048 \
    --fix_modality_adapter false \
    --include_text_fields true \
    --name_dropout 0.1 \
    --taxonomy_dropout 0.8 \
    --train_split "train" \
    --eval_split "eval"

# ============================================================================
# QUICK START: For instruction tuning only (no contrastive learning)
# ============================================================================

echo "Quick start: Instruction tuning only..."
echo "If you want to skip contrastive learning and go straight to instruction tuning:"

python scripts/train_instruct.py \
    --llama_path "Qwen/Qwen3-14B" \
    --root_csv_dir "./data" \
    --save_checkpoint_dir "./checkpoints/instruct_only" \
    --torch_dtype "float16" \
    --batch_size_per_device 2 \
    --num_epochs 3 \
    --save_every_epochs 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --scheduler_gamma 0.95 \
    --random_seed 42 \
    --lora_rank 16 \
    --adapter_intermediate_dim 2048 \
    --fix_modality_adapter false \
    --include_text_fields true \
    --name_dropout 0.1 \
    --taxonomy_dropout 0.8 \
    --train_split "train" \
    --eval_split "eval"

echo "Training complete!"
