import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import gc
import json
import os
import random
from datetime import datetime
import logging
from math import ceil

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TextStreamer,
)
from trl import SFTTrainer

from src.dataset.dataset import SOLDAugmentedDataset, SOLDDataset
from src.utils.helpers import get_device, save_checkpoint, setup_directories
from src.utils.logging_utils import setup_logging
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import is_bfloat16_supported

@dataclass
class ModelConfig:
    """Configuration for model training and inference"""
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: list = None

    def __post_init__(self):
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

class OffensiveLanguageDetector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._set_seed()
        self.logger = setup_logging()
        self.model_config = ModelConfig()
        self.device = get_device()
        self._setup_wandb()

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases tracking"""
        config = {
            "learning_rate": self.args.lr,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "model": self.args.pretrained_model,
            "seed": self.args.seed,
            "dataset": self.args.dataset,
            "using_augmented_dataset": self.args.use_augmented_dataset
        }
        
        wandb.init(
            project=self.args.wandb_project,
            config=config,
            name=f"{self.args.exp_name}_{'TRAIN' if not self.args.test else 'TEST'}"
        )
        self.args.wandb_run_url = wandb.run.get_url()

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load and configure the model and tokenizer"""
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.args.pretrained_model,
                max_seq_length=self.model_config.max_seq_length,
                dtype=self.model_config.dtype,
                load_in_4bit=self.model_config.load_in_4bit,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=self.model_config.lora_r,
                target_modules=self.model_config.target_modules,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.args.seed,
                use_rslora=False,
                loftq_config=None,
            )

            tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
            
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _prepare_dataset(self, tokenizer) -> Dataset:
        """Prepare and process the dataset"""
        try:
            dataset_cls = SOLDAugmentedDataset if self.args.use_augmented_dataset else SOLDDataset
            train_dataset = dataset_cls(self.args, 'train')
            train_dataloader = DataLoader(train_dataset, 
                                        batch_size=1, 
                                        shuffle=True)
            
            messages_list = []
            for batch in train_dataloader:
                text, label, rationale, tokens = batch[0][0], batch[1][0], batch[2], batch[3][0].split()
                
                # Process offensive phrases outside the f-string
                offensive_phrases = ''
                if label == 'OFF' and rationale:
                    # Initialize variables for phrase grouping
                    phrases = []
                    current_phrase = []
                    
                    # Iterate through rationale and tokens together
                    for i, (r, token) in enumerate(zip(rationale, tokens)):
                        if r == 1:
                            current_phrase.append(token)
                        elif current_phrase:  # r == 0 and we have accumulated tokens
                            phrases.append(" ".join(current_phrase))
                            current_phrase = []
                    
                    # Don't forget to add the last phrase if it exists
                    if current_phrase:
                        phrases.append(" ".join(current_phrase))
                    
                    offensive_phrases = f'Offensive Phrases: {", ".join(phrases)}' if phrases else ''
                    print()
                
                messages_list.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an emotionally intelligent assistant who speaks Sinhala and English Languages. Your task is to determine whether each tweet is OFFENSIVE or NOT OFFENSIVE. For each tweet, provide a single word as your output: either \"OFF\" or \"NOT\". And if the tweet is OFFENSIVE, provide a phrases in the tweet that you find offensive."
                        },
                        {
                            "role": "user", 
                            "content": f"determine whether the following Tweet is OFFENSIVE (OFF) or NOT OFFENSIVE (NOT): '{text}'"
                        },
                        {
                            "role": "assistant",
                            "content": f"{label}\n{offensive_phrases}"
                        }
                    ]
                })


            dataset = Dataset.from_list(messages_list)

            def formatting_prompts_func(examples):
                convos = examples["messages"]
                texts = [tokenizer.apply_chat_template(convo, 
                                                     tokenize=False, 
                                                     add_generation_prompt=False) 
                        for convo in convos]
                return {"text": texts}

            return dataset.map(formatting_prompts_func, batched=True)
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise

    def train(self) -> None:
        """Train the model"""
        try:
            model, tokenizer = self._load_model_and_tokenizer()
            dataset = self._prepare_dataset(tokenizer)

            random_samples = dataset.shuffle().select(range(5))
            for sample in random_samples:
                print(sample)
                print('\n')


            training_args = TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5, 
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps=10, # remove for full training run
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                output_dir=self.args.dir_result,
                report_to='wandb'
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.model_config.max_seq_length,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                dataset_num_proc=2,
                packing=False,
                args=training_args,
            )

            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            print(f"{start_gpu_memory} GB of memory reserved.")

            self._update_wandb_config({"start_gpu_memory": start_gpu_memory, "max_memory": max_memory})

            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            )

            trainer_stats = trainer.train()
            self._log_training_stats(trainer_stats)
            
            save_path, huggingface_repo_url = "", ""
            if self.args.push_to_hub:
                save_path, huggingface_repo_url = save_checkpoint(self.args, model, tokenizer)
            self._update_wandb_config({"checkpoint": save_path, 
                                     "huggingface_repo_url": huggingface_repo_url})

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.args.test:
                self.logger.info("Testing")
                self.test(model, tokenizer)
            else:
                wandb.finish()

    def _prepare_dataset_test(self, tokenizer) -> Dataset:
            """Prepare and process the dataset"""
            try:
                dataset_cls = SOLDAugmentedDataset if self.args.use_augmented_dataset else SOLDDataset
                train_dataset = dataset_cls(self.args, 'test')
                train_dataloader = DataLoader(train_dataset, 
                                            batch_size=1, 
                                            shuffle=True)
                
                messages_list = []
                for batch in train_dataloader:
                    text, label, rationale, tokens = batch[0][0], batch[1][0], batch[2], batch[3][0].split()
                
                    # Process offensive phrases outside the f-string
                    offensive_phrases = ''
                    phrases_only = ''
                    if label == 'OFF' and rationale:
                        # Initialize variables for phrase grouping
                        phrases = []
                        current_phrase = []
                        
                        # Iterate through rationale and tokens together
                        for i, (r, token) in enumerate(zip(rationale, tokens)):
                            if r == 1:
                                current_phrase.append(token)
                            elif current_phrase:  # r == 0 and we have accumulated tokens
                                phrases.append(" ".join(current_phrase))
                                current_phrase = []
                        
                        # Don't forget to add the last phrase if it exists
                        if current_phrase:
                            phrases.append(" ".join(current_phrase))
                        
                        offensive_phrases = f'Offensive Phrases: {", ".join(phrases)}' if phrases else ''
                        phrases_only = f"{', '.join(phrases)}" if phrases else ''
            
                    
                    messages_list.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an emotionally intelligent assistant who speaks Sinhala and English Languages. Your task is to determine whether each tweet is OFFENSIVE or NOT OFFENSIVE. For each tweet, provide a single word as your output: either \"OFF\" or \"NOT\". And if the tweet is OFFENSIVE, provide a phrases in the tweet that you find offensive."
                            },
                            {
                                "role": "user", 
                                "content": f"determine whether the following Tweet is OFFENSIVE (OFF) or NOT OFFENSIVE (NOT): '{text}'"
                            }
                    
                        ],
                        "actual_tweet": text,
                        "label": batch[1][0],
                        "rationale": batch[2],
                        "offensive_phrases": phrases_only,
                        "tokens": batch[3][0].split()
                    })


                dataset = Dataset.from_list(messages_list)

                def formatting_prompts_func(examples):
                    convos = examples["messages"]
                    texts = [tokenizer.apply_chat_template(convo, 
                                                        tokenize=False, 
                                                        add_generation_prompt=False) 
                            for convo in convos]
                    return {"text": texts}

                return dataset.map(formatting_prompts_func, batched=True)
            except Exception as e:
                self.logger.error(f"Error preparing dataset: {str(e)}")
                raise


    def test(self, model, tokenizer) -> None:
        """Test the model"""
        try:
            dataset = self._prepare_dataset_test(tokenizer)
            if model == None and self.args.test_model_path:
                model = FastLanguageModel.from_pretrained(self.args.test_model_path)
            
            FastLanguageModel.for_inference(model)
            true_labels = []
            predicted_labels = []
            predicted_offensive_phrases = []
            actual_tweets_list, offensive_phrases_list, rationale_list, tokens_list = [], [], [], []

            for test_sample in dataset.select(range(5)):
                actual_tweet = test_sample['actual_tweet']
                label = test_sample['label']
                rationale = test_sample['rationale']
                offensive_phrases = test_sample['offensive_phrases']
                tokens = test_sample['tokens']

                actual_tweets_list.append(actual_tweet)
                offensive_phrases_list.append(offensive_phrases)
                rationale_list.append(rationale)
                tokens_list.append(tokens)
                true_labels.append(label)
                
                input_ids = tokenizer.apply_chat_template(
                test_sample,
                add_generation_prompt = True,
                return_tensors = "pt",
                    ).to("cuda")
            
            
                text_streamer = TextStreamer(tokenizer, skip_prompt = True)
                gen = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)
                generated_answer = tokenizer.decode(gen[0], skip_special_tokens = False)
                # Extract the assistant response between the markers

                gen_label, gen_offensive_phrases = self._extract_components(generated_answer)
               
                
                predicted_labels.append(gen_label)
                predicted_offensive_phrases.append(gen_offensive_phrases)



            actual_binary = [1 if label.strip() == 'OFF' else 0 for label in true_labels]
            predicted_binary = [1 if label.strip() == 'OFF' else 0 for label in predicted_labels]

            clas_rprt = classification_report(actual_binary, predicted_binary, 
                             target_names=['NOT', 'OFF'])
            wandb.log({"classification_report": clas_rprt})

            # Create table data by zipping all lists together
            table_data = list(zip(true_labels, predicted_labels, actual_tweets_list, 
                                offensive_phrases_list, predicted_offensive_phrases, 
                                rationale_list, tokens_list))
            
            wandb_table_of_predictions = wandb.Table(
                columns=["True Labels", "Predicted Labels", "Actual Tweet", 
                        "Actual Offensive Phrases", "Predicted Offensive Phrases", 
                        "Rationale", "Tokens"], 
                data=table_data
            )
            wandb.log({"Table_of_predictions": wandb_table_of_predictions})
            
        except Exception as e:
            self.logger.error(f"Testing failed: {str(e)}")
            raise
        finally:
            wandb.finish()
    
    @staticmethod
    def _extract_components(text):
        # Find the last section after "assistant" header
        last_section = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        
        # Split the components by newline
        components = last_section.strip("<|eot_id|>").strip().split("\n\n")
        
        # Extract LBL and OFFENSIVE PHRASES LIST
        lbl = components[0]
        offensive_phrases = components[1] if len(components) > 1 else ""
        
        return lbl, offensive_phrases


    def _log_training_stats(self, trainer_stats: Dict[str, Any]) -> None:
        """Log training statistics"""
        gpu_stats = torch.cuda.get_device_properties(0)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        
        stats = {
            "training_time_seconds": trainer_stats.metrics['train_runtime'],
            "training_time_minutes": round(trainer_stats.metrics['train_runtime']/60, 2),
            "peak_memory_gb": used_memory,
            "peak_memory_percentage": round(used_memory/max_memory*100, 3)
        }
        
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
            wandb.log({key: value})

    def _update_wandb_config(self, config_updates: Dict[str, Any]) -> None:
        """Update W&B configuration"""
        wandb.config.update(config_updates, allow_val_change=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Offensive Language Detector')

    # SEED 
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # DATASET
    dataset_choices = ['sold', 'hatexplain']
    parser.add_argument('--dataset', default='sold', choices=dataset_choices, help='a dataset to use')

    # TESTING 
    parser.add_argument('--test', default=True, help='test the model', type=bool)
    parser.add_argument('--test_model_path', type=str, required=False, help='the checkpoint path to test', default="test/checkpoint-1001")

    # PRETRAINED MODEL
    model_choices = ['unsloth/Llama-3.2-3B-Instruct', 'unsloth/Llama-3.2-1B-Instruct-bnb-4bit', 
                    'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', 'unsloth/Mistral-Small-Instruct-2409']
    parser.add_argument('--pretrained_model', default='unsloth/Llama-3.2-3B-Instruct', 
                       choices=model_choices, help='a pre-trained LLM')  

    # TRAINING PARAMETERS
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00002, help='learning rate')
    
    # OUTPUT AND SAVING
    parser.add_argument('--dir_result', type=str, default='results', help='directory to save results')
    parser.add_argument('--exp_save_name', type=str, default=None, help='an experiment name')
    parser.add_argument('--short_name', default=False, help='use a shorter name for model checkpoints', type=bool)

    # WEIGHTS & BIASES CONFIG
    parser.add_argument('--wandb_project', type=str, default='subasa-llm', help='Weights & Biases project name')
    parser.add_argument('--push_to_hub', default=False, help='save the model to huggingface', type=bool)
    parser.add_argument('--huggingface_repo', type=str, help='HuggingFace repository name when pushing to hub')

    # DATASET AUGMENTATION
    parser.add_argument('--skip_empty_rat', default=False, help='skip empty rationales', type=bool, required=False)
    parser.add_argument('--use_augmented_dataset', default=False, help='use augmented dataset', type=bool)

    return parser.parse_args()

def main():
    args = parse_args()
    args.exp_date = datetime.now().strftime("%m%d-%H%M")
    
    # Setup paths and experiment name
    args.device = get_device()
    args.exp_name, args.dir_result = setup_directories(args)
    
    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    detector = OffensiveLanguageDetector(args)
    detector.train()

if __name__ == "__main__":
    main()


