from typing import List, Dict, Any, Optional
from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split

from src.utils.helpers import add_tokens_to_tokenizer
from src.utils.logging_utils import setup_logging

class BaseDataset(Dataset):
    def __init__(self, args, mode: str = 'train', tokenizer: Optional[XLMRobertaTokenizer] = None):
        self.train_dataset_path = 'SOLD_DATASET/sold_train_split.json'
        self.test_dataset_path = 'SOLD_DATASET/sold_test_split.json'
        self.label_list = ['NOT', 'OFF']
        self.label_count = [0, 0]
        self.logger = setup_logging()
        self.mode = mode
        self.dataset: List[Dict[str, Any]] = []
        
        # Load and initialize tokenizer if not provided
        self.tokenizer = tokenizer if tokenizer else self._initialize_tokenizer(args)
        
        # Load dataset based on mode
        self._load_dataset(args)

    def _initialize_tokenizer(self, args) -> XLMRobertaTokenizer:
        """Initialize and setup tokenizer"""
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
        return add_tokens_to_tokenizer(args, tokenizer)

    def _load_dataset(self, args) -> None:
        """Load and prepare dataset based on mode"""
        try:
            if self.mode == 'test':
                with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
                    self.dataset = list(json.load(f))
            elif self.mode in ['train', 'val']:
                with open(self.train_dataset_path, 'r', encoding='utf-8') as f:
                    full_dataset = list(json.load(f))
                
                # Split dataset for train/val
                train_set, val_set = train_test_split(
                    full_dataset, 
                    test_size=0.1, 
                    random_state=args.seed
                )
                
                self.dataset = train_set if self.mode == 'train' else val_set
                self._count_labels()

            # Sort dataset for consistency
            self.dataset.sort(key=lambda x: x['post_id'])

            # Optionally limit dataset size for debugging
            if hasattr(args, 'debug_mode') and args.debug_mode:
                self.dataset = self.dataset[:100]

        except FileNotFoundError as e:
            self.logger.error(f"Dataset file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in dataset file: {e}")
            raise

    def _count_labels(self) -> None:
        """Count occurrences of each label"""
        for item in self.dataset:
            for i, label in enumerate(self.label_list):
                if item['label'] == label:
                    self.label_count[i] += 1

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[Dict[str, str]]]:
        """Get a single item from the dataset with proper message formatting"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an emotionally intelligent assistant who speaks Sinhala and English Languages. Your task is to determine whether each tweet is OFFENSIVE or NOT OFFENSIVE. For each tweet, provide a single word as your output: either \"OFF\" or \"NOT\"."
                },
                {
                    "role": "user", 
                    "content": f"determine whether the following Tweet is OFFENSIVE (OFF) or NOT OFFENSIVE (NOT): '{self.dataset[idx]['text']}'"
                },
                {
                    "role": "assistant",
                    "content": f"{self.dataset[idx]['label']}"
                }
            ]
        }
