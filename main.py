# Standard library imports 
import argparse
import gc
import json
import os
import random 
from datetime import datetime
from math import ceil
from unsloth import FastLanguageModel
import torch

# Third party imports
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
import wandb
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    DataCollatorForLanguageModeling,
)

# Local imports
from src.dataset.dataset import SOLDAugmentedDataset, SOLDDataset
from src.evaluate.evaluate import evaluate, evaluate_for_hatespeech
from src.evaluate.lime import TestLime
from src.models.custom_models import XLMRobertaCustomForTCwMRP
from src.utils.helpers import (
    GetLossAverage,
    NumpyEncoder, 
    add_tokens_to_tokenizer,
    get_device,
    save_checkpoint,
    setup_directories
)
from src.utils.logging_utils import setup_logging
from src.utils.prefinetune_utils import add_pads, make_masked_rationale_label, prepare_gts
import subprocess
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Subasa - Adapting Language Models for Low Resourced Offensive Language Detection in Sinhala')

    #SEED 
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # DATASET
    dataset_choices = ['sold', 'hatexplain']
    parser.add_argument('--dataset', default='sold', choices=dataset_choices, help='a dataset to use')

    # TESTING 
    parser.add_argument('--test', default=False, help='test the model', type=bool)
    parser.add_argument('--test_model_path', type=str, required=False, help='the checkpoint path to test', default=None)

    # PRETRAINED MODEL
    model_choices = ['unsloth/Llama-3.2-3B-Instruct', 'unsloth/Llama-3.2-1B-Instruct-bnb-4bit', 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', 'unsloth/Mistral-Small-Instruct-2409' ]
    parser.add_argument('--pretrained_model', default='unsloth/Llama-3.2-3B-Instruct', choices=model_choices, help='a pre-trained LLM')  

    # TRAIN
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--val_int', type=int, default=945)  
    parser.add_argument('--patience', type=int, default=3)
    
    parser.add_argument('--skip_empty_rat', default=False, help='skip empty rationales', type=bool, required=False)

    # Weights & Biases config
    parser.add_argument('--wandb_project', type=str, default='subasa-llm', help='Weights & Biases project name')
    parser.add_argument('--push_to_hub', default=False, help='save the model to huggingface', type=bool)
    
    parser.add_argument('--num_labels', type=int, default=2) # number of classes in the dataset

    ## Explainability based metrics
    parser.add_argument('--explain_sold', default=False, help='Generate Explainablity Metrics', type=bool)
    parser.add_argument('--top_k', default=5, help='the top num of attention values to evaluate on explainable metrics')
    parser.add_argument('--lime_n_sample', default=100, help='the num of samples for lime explainer')

    ## User given experiment name
    parser.add_argument('--exp_save_name', type=str, default=None, help='an experiment name')

    ## Use a shorter file name for model checkpoints
    parser.add_argument('--short_name', default=False, help='use a shorter name for model checkpoints', type=bool)

    #TEMP Skip arg for testing data augmentation
    parser.add_argument('--skip', default=False, help='skip data augmentation', type=bool)


    # Use Augmented Dataset
    parser.add_argument('--use_augmented_dataset', default=False, help='use augmented dataset', type=bool)


    return parser.parse_args()




def train_offensive_detection(args):
    # Setup logging
    logger = setup_logging()
    logger.info("[START] [LLM] [FINAL] [OFFENSIVE_LANG_DET] Starting with args: {}".format(args))

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "model": args.pretrained_model,
            "seed": args.seed,
            "dataset": args.dataset,
            "val_int": args.val_int,
            "patience": args.patience,
            "label_classes": args.num_labels,
            "skip_empty_rat": args.skip_empty_rat,
            "test": args.test,
            "explain_sold": args.explain_sold,
        },
        name=args.exp_name + '_TRAIN'
    )

    args.wandb_run_url = wandb.run.get_url()

    # Set seed
    set_seed(args.seed)

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =  args.pretrained_model, #"unsloth/Llama-3.2-3B-Instruct",
        #model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

    # Define dataloader
    if args.use_augmented_dataset == False:
        train_dataset = SOLDDataset(args, 'train') 
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_dataset = SOLDAugmentedDataset(args, 'train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    from datasets import Dataset
    list_ds = []

    # create a HF Dataset using train_dataloader
    for i, batch in enumerate(train_dataloader):
        list_ds.append(batch)
    
    dataset = Dataset.from_dict(list_ds)

    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True,)




    log = open(os.path.join(args.dir_result, 'train_res.txt'), 'a')

    steps_per_epoch = ceil(len(train_dataset) / args.batch_size)
    print("Steps per epoch: ", steps_per_epoch)

    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 3000,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
            report_to = 'wandb'
        ),
    )

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    wandb.config.update({
        "using_augmented_dataset": args.use_augmented_dataset,
        "train_dataset_size": len(train_dataset),
        "steps_per_epoch": steps_per_epoch,
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "val_int": args.val_int,
        
        "gpu_start_memory": start_gpu_memory,
    })

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



    save_path, huggingface_repo_url = save_checkpoint(model)

    #update wandb config with the huggingface repo url and save path of checkpoint
    wandb.config.update({
        "checkpoint": save_path,
        "huggingface_repo_url": huggingface_repo_url,
    }, allow_val_change=True)

    log.close()
    wandb.finish()


def test_for_hate_speech(args):
    set_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        config={
            "batch_size": args.batch_size,
            "model": args.pretrained_model,
            "test_model_path": args.test_model_path,
            "seed": args.seed,
            "dataset": args.dataset,
            "val_int": args.val_int,
            "patience": args.patience,
            "top_k": args.top_k,
            "lime_n_sample": args.lime_n_sample,
            "label_classes": args.num_labels,
            "test": args.test,
            "exp_name": args.exp_name,
            "explain_sold": args.explain_sold,
        },
        name=args.exp_name + '_TEST'
    )

        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    messages = [
        {'content': 'You are a data extraction tool. Return answers in JSON format only.',
    'role': 'system'},
    {'content': "Identify the project names, company names, and people names in the "+\
    "'following highlight: 'Together with our partners from Nike, Justin Chanell completed"+\
    "' the milestone for GenXneG, which accelerated Nvidia's revenue by 5% and Tesla's revenue "+\
    "'by 3%, the media went nuts over these accomplishments'",
    'role': 'user'}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)


if __name__ == '__main__':
    args = parse_args()
    args.device = get_device()
    args.waiting = 0
    args.n_eval = 0

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Setup experiment name and paths
    lm = '-'.join(args.pretrained_model.split('-')[:])
    now = datetime.now()
    args.exp_date = (now.strftime('%d%m%Y-%H%M') + '_LK')

    # Setup experiment name and directories
    args.exp_name, args.dir_result = setup_directories(args)
    print("Checkpoint path: ", args.dir_result)

    if args.test and args.test_model_path:
        args.explain_sold = False  # Turn it to True for explainable metrics | WIP
        args.batch_size = 1
        test_for_hate_speech(args)
    elif not args.test:
        train_offensive_detection(args)


