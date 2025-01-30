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

from sklearn.metrics import classification_report
from src.dataset.dataset import SOLDAugmentedDataset, SOLDDataset
from src.utils.helpers import get_device, save_checkpoint, setup_directories
from src.utils.logging_utils import setup_logging
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import is_bfloat16_supported
# from src.utils.env import load_env_variables

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
            name=f"{self.args.exp_name}"
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

    def _get_model_type(self) -> str:
        """Determine the model type from the model name"""
        if "mistral" in self.args.pretrained_model.lower():
            return "mistral"
        return "llama"

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

            model_type = self._get_model_type()
            if model_type == "mistral":
                tokenizer = get_chat_template(tokenizer, chat_template="mistral")
            else:
                tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
            
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _process_offensive_phrases(self, label: str, rationale: list, tokens: list) -> Tuple[str, str]:
        """Process and extract offensive phrases from text"""
        offensive_phrases = ''
        phrases_only = ''
        if label == 'OFF' and rationale:
            phrases = []
            current_phrase = []
            
            for r, token in zip(rationale, tokens):
                if r == 1:
                    current_phrase.append(token)
                elif current_phrase:
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
            
            if current_phrase:
                phrases.append(" ".join(current_phrase))
            
            offensive_phrases = f'Offensive Phrases: {", ".join(phrases)}' if phrases else ''
            phrases_only = ', '.join(phrases) if phrases else ''
        
        return offensive_phrases, phrases_only

    def _create_message(self, text: str, label: str = None, offensive_phrases: str = '', is_testing: bool = False) -> dict:
        """Create a message dictionary for the model"""
        model_type = self._get_model_type()

        system_msg = """You are an emotionally intelligent assistant who speaks Sinhala and English Languages. Your task is to determine whether each tweet is OFFENSIVE or NOT OFFENSIVE. For each tweet, provide a single word as your output: either \"OFF\" or \"NOT\". For offensive tweets, identify and list the specific offensive phrases without translation.\n"""

        user_msg = f"Please classify the following tweet as \"OFF\" or \"NOT\". If offensive, list the specific offensive phrases:\n\n'{text}'"

        
        if model_type == "mistral":
            # system_msg = "You are an emotionally intelligent assistant who speaks Sinhala and English Languages. Your task is to determine whether each tweet is OFFENSIVE or NOT OFFENSIVE. For each tweet, provide a single word as your output: either \"OFF\" or \"NOT\". And if the tweet is OFFENSIVE, provide phrases in the tweet that you find offensive. You should only explicityl state the offensive phrases and not give any kind of translation\n"
            # user_msg = f"Determine whether the following Tweet is OFFENSIVE (OFF) or NOT OFFENSIVE (NOT): '{text}'"
            if not is_testing and label:
                assistant_msg = f"{label}\n{offensive_phrases}"
        else:
            # system_msg = "You are an emotionally intelligent assistant who speaks Sinhala and English Languages. Your task is to determine whether each tweet is OFFENSIVE or NOT OFFENSIVE. For each tweet, provide a single word as your output: either \"OFF\" or \"NOT\". And if the tweet is OFFENSIVE, provide phrases in the tweet that you find offensive."
            # user_msg = f"determine whether the following Tweet is OFFENSIVE (OFF) or NOT OFFENSIVE (NOT): '{text}'"
            if not is_testing and label:
                assistant_msg = f"{label}\n{offensive_phrases}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        if not is_testing and label:
            messages.append({"role": "assistant", "content": assistant_msg})
        
        return {"messages": messages}

    def _prepare_dataset(self, tokenizer, mode: str = 'train') -> Dataset:
        """Prepare and process the dataset for training or testing"""
        try:
            dataset_cls = SOLDDataset if mode == 'test' else (SOLDAugmentedDataset if self.args.use_augmented_dataset else SOLDDataset)
            dataset = dataset_cls(self.args, mode)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            messages_list = []
            for batch in dataloader:
                text, label = batch[0][0], batch[1][0]
                rationale, tokens = batch[2], batch[3][0].split()
                
                offensive_phrases, phrases_only = self._process_offensive_phrases(label, rationale, tokens)
                if mode == 'train':
                    message_data = self._create_message(text, label, offensive_phrases)
                elif mode == 'test':
                    message_data = self._create_message(text, label, offensive_phrases, is_testing=True)
                
                if mode == 'test':
                    message_data.update({
                        "actual_tweet": text,
                        "label": label,
                        "rationale": rationale,
                        "offensive_phrases": phrases_only,
                        "tokens": tokens
                    })
                
                messages_list.append(message_data)

            return self._format_dataset(messages_list, tokenizer)
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise

    def _format_dataset(self, messages_list: list, tokenizer) -> Dataset:
        """Format the dataset for the model"""
        dataset = Dataset.from_list(messages_list)
        
        def formatting_prompts_func(examples):
            dls = [
                    tokenizer.apply_chat_template(
                        convo, 
                        tokenize=False, 
                        add_generation_prompt=False
                    ) for convo in examples["messages"]
                ]
            return {
                "text": dls
            }
        
        return dataset.map(formatting_prompts_func, batched=True)

    def train(self) -> None:
        """Train the model"""
        try:
            model, tokenizer = self._load_model_and_tokenizer()
            dataset = self._prepare_dataset(tokenizer)

            random_samples = dataset.shuffle().select(range(1))
            for sample in random_samples:
                print(sample)
                print('\n')

            from datetime import datetime

            now = datetime.now()
            time_now = (now.strftime('%d%m%Y-%H%M'))


            training_args = TrainingArguments(
                per_device_train_batch_size=self.args.per_device_train_batch_size,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                warmup_steps=self.args.warmup_steps,
                # num_train_epochs=self.args.epochs if not self.args.max_steps else None,
                # max_steps=self.args.max_steps,
                learning_rate=self.args.lr,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                # logging_steps=self.args.logging_steps,
                optim="adamw_8bit",
                weight_decay=self.args.weight_decay,
                lr_scheduler_type="linear",
                output_dir=self.args.dir_result,
                report_to='wandb',
                run_name=self.args.exp_name + '_' + time_now,
            )

            if self.args.max_steps:
                training_args.max_steps = self.args.max_steps
            else:
                training_args.num_train_epochs = self.args.epochs

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

            model_type = self._get_model_type()
        
            if model_type == "llama":
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
                )
            elif model_type == "mistral":
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="[INST]" , #"<|user|>",
                    response_part= "[/INST]"  #"<|assistant|>",
                )

            trainer_stats = trainer.train()
            self._log_training_stats(trainer_stats)
            
            # save_path, huggingface_repo_url = "", ""
            # if self.args.push_to_hub:
            #     save_path, huggingface_repo_url = save_checkpoint(self.args, model, tokenizer)
            # self._update_wandb_config({"checkpoint": save_path, 
            #                          "huggingface_repo_url": huggingface_repo_url})

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.args.test:
                self.logger.info("Testing")
                self.test(model, tokenizer)
            else:
                wandb.finish()

    def evaluate(self) -> None:
        """Evaluate a pre-trained model from HuggingFace"""
        try:
            if not self.args.hf_model_path:
                raise ValueError("HuggingFace model path must be provided for evaluation")

            self.logger.info(f"Loading model from {self.args.hf_model_path}")
            model, tokenizer = self._load_model_and_tokenizer()
            
            self.logger.info("Starting evaluation")
            self.test(model, tokenizer)

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    def test(self, model=None, tokenizer=None) -> None:
        """Test the model"""
        try:
            if model is None or tokenizer is None:
                if not self.args.hf_model_path:
                    raise ValueError("Either provide model and tokenizer or specify hf_model_path")
                model, tokenizer = self._load_model_and_tokenizer()

            dataset = self._prepare_dataset(tokenizer, mode='test')

            if self.args.debug:
                dataset = dataset.select(range(5))
            
            start_time = datetime.now()
            FastLanguageModel.for_inference(model)
            true_labels = []
            predicted_labels = []
            predicted_offensive_phrases = []
            actual_tweets_list, offensive_phrases_list, rationale_list, tokens_list = [], [], [], []

            length_of_dataset = len(dataset)
            counter = 0

            model_type = self._get_model_type()
            for test_sample in dataset: #TODO remove the range after testing
                counter += 1
                print(f"Processing sample {counter}/{length_of_dataset} \n")

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
                test_sample['messages'],
                add_generation_prompt = True,
                return_tensors = "pt",
                    ).to("cuda")
            
                text_streamer = TextStreamer(tokenizer, skip_prompt = True)
                gen = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)
                generated_answer = tokenizer.decode(gen[0], skip_special_tokens = False)

                gen_label, gen_offensive_phrases = self._extract_components(generated_answer, model_type)
                
                predicted_labels.append(gen_label)
                predicted_offensive_phrases.append(gen_offensive_phrases)

            actual_binary = [1 if label.strip() == 'OFF' else 0 for label in true_labels]
            predicted_binary = [1 if label.strip() == 'OFF' else 0 for label in predicted_labels]

            clas_rprt = classification_report(actual_binary, predicted_binary, 
                             target_names=['NOT', 'OFF'], output_dict=True)
            wandb.log({"classification_report": clas_rprt})
            end_time = datetime.now()
            duration = end_time - start_time
            wandb.log({"testing_duration": duration.total_seconds()})

            # Create table data
            table_data = [
                {
                    "True Labels": tl,
                    "Predicted Labels": pl,
                    "Actual Tweet": at,
                    "Actual Offensive Phrases": aop,
                    "Predicted Offensive Phrases": pop,
                    "Rationale": rat,
                    "Tokens": tok
                }
                for tl, pl, at, aop, pop, rat, tok in zip(
                    true_labels, predicted_labels, actual_tweets_list,
                    offensive_phrases_list, predicted_offensive_phrases,
                    rationale_list, tokens_list
                )
            ]

            # Save to JSON with UTF-8 encoding
            json_path = os.path.join(self.args.dir_result, "predictions.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, ensure_ascii=False, indent=2)

            # Log JSON file to wandb
            wandb.save(json_path)

            # Also create wandb table
            wandb_table = wandb.Table(
                columns=["True Labels", "Predicted Labels", "Actual Tweet", 
                        "Actual Offensive Phrases", "Predicted Offensive Phrases", 
                        "Rationale", "Tokens"], 
                data=list(zip(true_labels, predicted_labels, actual_tweets_list, 
                            offensive_phrases_list, predicted_offensive_phrases, 
                            rationale_list, tokens_list))
            )
            wandb.log({"Table_of_predictions": wandb_table})
            
        except Exception as e:
            self.logger.error(f"Testing failed: {str(e)}")
            raise
        finally:
            wandb.finish()
    
    @staticmethod
    def _extract_components(text, model_type="llama"):
        """Extract label and offensive phrases based on model type"""
        if model_type == "mistral":
            # Extract content between <|assistant|> and </s>
            assistant_content = text.split("[/INST]")[-1].split("</s>")[0].strip()
            assistant_content = assistant_content.split("\n")
            # remove newline characters in assistant_content array
            # if assistant_content[1]:
            #     off_phrases = assistant_content[1].split(":")[-1]
            #     assistant_content[1] = off_phrases
        else:
            # Extract content after assistant header for Llama
            assistant_content = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip("<|eot_id|>").strip()
            assistant_content = assistant_content.split("\n")
        
        # Split components by newline
        # components = assistant_content.strip().split("\n\n")
        
        # Extract label and offensive phrases
        lbl = assistant_content[0]
        offensive_phrases = assistant_content[1:]
        offensive_phrases = " ".join(offensive_phrases) if offensive_phrases else ""
        
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
    parser.add_argument('--hf_model_path', type=str, required=False, help='the checkpoint path to test', default=None)

    # PRETRAINED MODEL
    model_choices = [
        'unsloth/Llama-3.2-3B-Instruct', 
        'unsloth/Llama-3.2-1B-Instruct-bnb-4bit',
        'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', 
        'unsloth/Mistral-Small-Instruct-2409',
        'unsloth/mistral-7b-instruct-v0.3-bnb-4bit'
    ]
    parser.add_argument('--pretrained_model', default='unsloth/Llama-3.2-3B-Instruct', 
                       choices=model_choices, help='a pre-trained LLM')  

    # TRAINING PARAMETERS
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--warmup_steps', type=int, default=5, help='number of warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='number of gradient accumulation steps')
    parser.add_argument('--max_steps', type=int, default=None, help='maximum number of training steps (overrides epochs if set)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for optimizer')
    parser.add_argument('--logging_steps', type=int, default=1, help='number of steps between logging')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='batch size per device during training')

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

    #DEBUG
    parser.add_argument('--debug', default=False, help='debug mode', type=bool)

    return parser.parse_args()

def main():
    # Load environment variables
    # load_env_variables()
    
    args = parse_args()
    args.exp_date = datetime.now().strftime("%m%d-%H%M")
    
    # Setup paths and experiment name
    args.device = get_device()
    args.exp_name, args.dir_result = setup_directories(args)
    
    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    detector = OffensiveLanguageDetector(args)
    if args.test and args.hf_model_path:
        detector.evaluate()
    else:
        detector.train()

if __name__ == "__main__":
    main()


