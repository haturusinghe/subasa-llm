# Subasa LLM

A task-specific fine-tuning framework for large language models (LLMs) focused on offensive language detection in multilingual contexts, with evaluation on the SOLD benchmark.

## Description

This project provides a specialized fine-tuning pipeline for LLMs (including Llama and Mistral variants) using LoRA (Low-Rank Adaptation) to optimize model performance on offensive language detection tasks. It features comprehensive support for Sinhala language, integrating seamlessly with the SOLD (Sinhala Offensive Language Detection) benchmark for evaluation.

## Features

- Sinhala offensive language detection 
- Support for multiple LLM architectures (Llama, Mistral)
- LoRA-based fine-tuning for efficient training
- Offensive phrase extraction and analysis
- Integration with Weights & Biases for experiment tracking
- Support for both standard and augmented datasets
- Comprehensive evaluation metrics and reporting
- Advanced data augmentation strategies:
  - Noun-based insertions
  - Adjective replacements
  - Verb modifications
  - Proper noun handling
  - Compound verb modifications
  - Case marker insertions
  - Context-aware punctuation patterns

## Data Augmentation

The project includes a sophisticated data augmentation system that:

- Analyzes POS (Part-of-Speech) patterns in existing offensive content
- Generates new offensive samples using linguistic patterns
- Supports multiple augmentation strategies:
  - Inserts offensive phrases in grammatically appropriate positions
  - Modifies existing phrases while maintaining syntactic structure
  - Handles compound verbs and case markers specific to Sinhala
  - Preserves linguistic validity through POS-aware modifications

### Augmentation Control Parameters

```bash
python main.py \
    --use_augmented_dataset True \
    --dataset "sold" \
    --pretrained_model "unsloth/Llama-3.2-3B-Instruct"
```

## Installation

```bash
# Clone the repository
git clone https://github.com/haturusinghe/subasa-llm.git

# Navigate to the project directory
cd subasa-llm

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py \
    --pretrained_model "unsloth/Llama-3.2-3B-Instruct" \
    --batch_size 16 \
    --epochs 1 \
    --lr 2e-4 \
    --wandb_project "subasa-llm"
```

### Evaluation

```bash
python main.py \
    --test True \
    --hf_model_path "path/to/your/model" \
    --dataset "sold"
```

### Key Parameters

- `--pretrained_model`: Choice of base model (Llama or Mistral variants)
- `--dataset`: Dataset choice ('sold' or 'hatexplain')
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--test`: Enable testing mode
- `--use_augmented_dataset`: Use augmented dataset version

## Model Configuration

The system uses the following default configurations:
- Maximum sequence length: 2048
- LoRA r: 16
- LoRA alpha: 16
- 4-bit quantization support
- Gradient checkpointing for memory efficiency

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

