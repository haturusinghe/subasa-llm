#!/bin/bash

# Your Hugging Face username
HF_USERNAME="your_username"

# Directory to save the downloaded models and results
DOWNLOAD_DIR="$HOME/huggingface_models"
RESULTS_DIR="$HOME/mmlu_results"

# Create directories if they don't exist
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$RESULTS_DIR"

# Install required packages
pip install lm-evaluation-harness
pip install torch
pip install "lm_eval[wandb]"

# Login to wandb (you'll need to do this once)
wandb login

# List of models to download and evaluate
MODELS=(
    "model1"
    "model2"
    "model3"
)

# Function to evaluate a model using MMLU
evaluate_model() {
    local model_path="$1"
    local model_name="$2"
    
    echo "Evaluating $model_name with MMLU benchmark..."
    
    # Run MMLU evaluation with wandb logging
    lm_eval \
        --model llama \
        --model_args "pretrained=$model_path" \
        --tasks mmlu \
        --num_fewshot 5 \
        --device cuda \
        --output_path "$RESULTS_DIR/${model_name}" \
        --log_samples \
        --wandb_args "project=mmlu-evaluation,name=$model_name,entity=your_wandb_entity" \
        --batch_size 8
}

# Loop through models and process each
for model in "${MODELS[@]}"; do
    echo "Processing $model..."
    
    # Download the model
    echo "Downloading $model..."
    huggingface-cli download "$HF_USERNAME/$model" --local-dir "$DOWNLOAD_DIR/$model"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $model"
        
        # Evaluate the model
        evaluate_model "$DOWNLOAD_DIR/$model" "$model"
        echo "Evaluation completed for $model"
        
        # Remove the downloaded model
        echo "Removing downloaded model..."
        rm -rf "$DOWNLOAD_DIR/$model"
        echo "Model removed successfully"
    else
        echo "Failed to download $model"
    fi
    echo "------------------------"
done

echo "All downloads and evaluations completed!"
