#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================="
echo "   LamSonOCR — Linux Setup & Train Script    "
echo "   (Optimized for Nvidia GPU / CUDA)         "
echo "============================================="

# 1. Check Python installation
echo "🔍 Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3 and try again."
    exit 1
fi
python3 --version

# 2. Setup virtual environment
echo "⚙️ Setting up virtual environment..."
if [ ! -f "venv/bin/python" ]; then
    rm -rf venv
    python3 -m venv venv
    echo "✅ Virtual environment 'venv' created."
else
    echo "ℹ️ Virtual environment 'venv' already exists. Skipping creation."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install requirements
echo "📦 Installing dependencies..."
pip install --upgrade pip

# Note: Default pip install for torch on Linux downloads CUDA-enabled binaries
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Installing pytorch and pillow directly..."
    pip install torch torchvision pillow tqdm pytest
fi
pip install bitstring

# Create directory structures
mkdir -p checkpoints data/all_train

# 3. Generate Synthetic Data
echo "📊 Generating synthetic dataset..."
# Generate 5,000 synthetic samples for training and 1,000 for validation
python main.py generate --output data/train --num-samples 5000
python main.py generate --output data/val --num-samples 1000

# 4. Extract ETL Datasets (if present)
echo "📂 Checking for ETL binary datasets..."

# Helper function to unzip if needed
unzip_if_needed() {
    local zip_file=$1
    local target_dir=$2
    if [ -f "$zip_file" ]; then
        if [ ! -d "$target_dir" ]; then
            echo "📦 Unzipping $zip_file..."
            python3 -c "import zipfile; zipfile.ZipFile('$zip_file').extractall('ETL')"
            echo "✅ Extracted $zip_file"
        else
            echo "ℹ️ $target_dir already exists. Skipping unzip."
        fi
    fi
}

unzip_if_needed "ETL/ETL1.zip" "ETL/ETL1"
unzip_if_needed "ETL/ETL2.zip" "ETL/ETL2"
unzip_if_needed "ETL/ETL3.zip" "ETL/ETL3"
unzip_if_needed "ETL/ETL4.zip" "ETL/ETL4"
unzip_if_needed "ETL/ETL5.zip" "ETL/ETL5"
unzip_if_needed "ETL/ETL6.zip" "ETL/ETL6"
unzip_if_needed "ETL/ETL7.zip" "ETL/ETL7"
unzip_if_needed "ETL/ETL8B.zip" "ETL/ETL8B"

HAS_ETL=false

# Check ETL4 (Hiragana)
if [ -f "ETL/ETL4/ETL4C" ]; then
    echo "🇯🇵 Found ETL4C. Extracting Hiragana character images..."
    python scripts/prepare_etl.py --input ETL/ETL4/ETL4C --output data/etl_train
    HAS_ETL=true
fi

# Check ETL3 (Alphanumeric/Katakana)
if [ -f "ETL/ETL3/ETL3C_1" ]; then
    echo "🔢 Found ETL3C_1. Extracting Alphanumeric character images..."
    python scripts/prepare_etl.py --input ETL/ETL3/ETL3C_1 --output data/etl_train
    HAS_ETL=true
fi

if [ -f "ETL/ETL3/ETL3C_2" ]; then
    echo "🔢 Found ETL3C_2. Extracting Alphanumeric character images..."
    python scripts/prepare_etl.py --input ETL/ETL3/ETL3C_2 --output data/etl_train
    HAS_ETL=true
fi

# 5. Merge Datasets
echo "🔄 Merging all available datasets..."
if [ -f "scripts/merge_datasets.py" ]; then
    python scripts/merge_datasets.py
else
    # Fallback merge if script is missing
    echo "image,text" > data/all_train/labels.csv
    for csv_file in data/train/labels.csv data/val/labels.csv data/etl_train/labels.csv; do
        if [ -f "$csv_file" ]; then
            tail -n +2 "$csv_file" >> data/all_train/labels.csv
        fi
    done
fi

# 6. Execute Training & Testing
echo "🧪 Running pytest suite to verify code correctness..."
PYTHONPATH=. venv/bin/python -m pytest tests/ -v



echo "🏋️ Starting OCR Model Training..."
# Auto-detect best device for training and evaluation
DEVICE=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "🖥️ Target hardware device: $DEVICE"
if [ "$DEVICE" = "cuda" ]; then
    python3 -c "import torch; print('   GPU Device Name:', torch.cuda.get_device_name(0))"
fi

python main.py train --train-data data/all_train --epochs 50 --batch-size 64

# 7. Execute Benchmark / Evaluation
echo "📋 Automatically Running Benchmark & Evaluation..."
python main.py evaluate --checkpoint checkpoints/best_model.pt --data data/val --device $DEVICE

echo "============================================="
echo "🎉 Setup, Training, and Benchmarking complete!"
echo "Model checkpoint saved to: checkpoints/best_model.pt"
echo "============================================="

