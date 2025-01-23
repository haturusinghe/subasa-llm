import json
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import os  
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,XLMRobertaForTokenClassification, XLMRobertaForSequenceClassification

from huggingface_hub import whoami

from src.models.custom_models import XLMRobertaCustomForTCwMRP
from src.utils.logging_utils import setup_logging

from huggingface_hub import HfApi
from huggingface_hub import login
from huggingface_hub import ModelCard, ModelCardData

def get_device():
    if torch.cuda.is_available():
        print("device = cuda")
        return torch.device('cuda')
    else:
        print("device = cpu")
        return torch.device('cpu')


def save_checkpoint(args, model, tokenizer):
    """
    Save model checkpoint locally and/or to HuggingFace Hub with different quantization options.
    Args:
        args: Arguments containing experiment configuration
        model: The trained model to save
        tokenizer: The tokenizer used with the model
    Returns:
        tuple: (local_save_path, huggingface_repo_url)
    """
    repo_id = f"haturusinghe/{args.exp_save_name}"
    save_path = None
    huggingface_repo_url = None

    # Define quantization configurations
    quantization_configs = {
        'default': {'method': None, 'enabled': False},
        'f16': {'method': "f16", 'enabled': True},
        'q4_k_m': {'method': "q4_k_m", 'enabled': False},
        'q8_0': {'method': None, 'enabled': False}
    }

    # Push to Hugging Face Hub
    huggingface_repo_url = None
    # Check if the HF_TOKEN environment variable is set
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        user = whoami(token=hf_token)
    else:
        print("No Hugging Face token found in environment variables. Please log in.")
        login()
    
    try:
        now = datetime.now()
        time_now = (now.strftime('%d%m%Y-%H%M'))
        
        for config_name, config in quantization_configs.items():
            if config['enabled']:
                huggingface_repo_url = repo_id
                model.push_to_hub_gguf(
                    repo_id,
                    tokenizer,
                    quantization_method=config['method'],
                    token=""  # Add your token here
                )


        markdown_file_save_path = os.path.join(args.dir_result, 'README.md')


        with open(markdown_file_save_path, 'w') as f:
            f.write(f"Wandb Run URL: {args.wandb_run_url}\n\n")
            
            f.write(f"""### Trainer Arguments:
                "learning_rate": {args.lr}
                "epochs": {args.epochs}
                "batch_size": {args.batch_size}
                "model": {args.pretrained_model}         
                "seed": {args.seed}
                "dataset": {args.dataset}
                "skip_empty_rat": {args.skip_empty_rat} """)
    
        # Create model card
        card_data = ModelCardData(
            language="en",
            license="mit",
            model_name=args.exp_save_name,
            base_model=args.pretrained_model,
        )

        card = ModelCard.from_template(
            card_data = card_data,
            template_path = markdown_file_save_path
        )

        
        
        # Push model card to hub
        card.push_to_hub(f"s-haturusinghe/{args.exp_save_name}")
        
        huggingface_repo_url = f"https://huggingface.co/s-haturusinghe/{args.exp_save_name}"
    
    except Exception as e:
            print(f"Error pushing to Hugging Face Hub: {str(e)}")
            print("Continuing without pushing to hub...")
    

    return save_path, huggingface_repo_url

def add_tokens_to_tokenizer(args, tokenizer):

    special_tokens_dict = {'additional_special_tokens': 
                            ['@USER', '<URL>']}  
    n_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.all_special_tokens) 
    # print(tokenizer.all_special_ids)
    
    return tokenizer

def setup_experiment_name(args):
    lm = '-'.join(args.pretrained_model.split('-')[:])
    name = f"{args.exp_date}_{args.lr}_{args.batch_size}_{args.val_int}_seed{args.seed}"
    
    
    return name

def setup_directories(args):
    if args.test:
        # Extract experiment name from test model path
        exp_name = args.test_model_path.split('/')[-2]
        base_dir = os.path.join("_finetune", exp_name)
        result_dir = os.path.join(base_dir, 'test')
    elif args.exp_save_name:
        exp_name = args.exp_save_name
        result_dir = os.path.join("_finetune", exp_name)
    else:
        exp_name = setup_experiment_name(args)
        result_dir = os.path.join("_finetune", exp_name)
    
    os.makedirs(result_dir, exist_ok=True)
    return exp_name, result_dir


