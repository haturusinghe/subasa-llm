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


def save_checkpoint(trained_model):

        # Save to 8bit Q8_0
    if False: trained_model.save_pretrained_gguf("model", tokenizer,)
    # Remember to go to https://huggingface.co/settings/tokens for a token!
    # And change hf to your username!
    if False: trained_model.push_to_hub_gguf("hf/model", tokenizer, token = "")

    # Save to 16bit GGUF
    if False: trained_model.save_pretrained_gguf("model2", tokenizer, quantization_method = "f16")
    if True: trained_model.push_to_hub_gguf("shadicopty/Llama3.2-3b-taxadvisor", tokenizer, quantization_method = "f16", token = "")

    # Save to q4_k_m GGUF
    if False: trained_model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    if False: trained_model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")
    
    return save_path, huggingface_repo_url


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


