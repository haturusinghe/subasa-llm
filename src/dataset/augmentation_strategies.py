from typing import List, Set, Dict, Tuple
import random

class AugmentationStrategies:
    def __init__(self, offensive_lexicon: Dict):
        self.offensive_lexicon = offensive_lexicon
        self.strategies = [
            self._handle_noun_insertions,
            self._handle_adjective_replacement,
            self._handle_verb_modification,
            self._handle_proper_noun_modification,
            self._handle_hybrid_approach,
            self._handle_punctuation_noun_pattern,
            self._handle_compound_verb_modification,
            self._handle_case_marker_insertion,
            self._handle_proper_noun_punctuation
        ]

    def apply_random_strategy(self, tokens: List[str], trigram: Tuple, i: int,
                            modified_tokens: List[str], inserted_positions: Set[int]) -> Tuple[List[str], Set[int], int]:
        """Apply a randomly selected augmentation strategy"""
        strategy = random.choice(self.strategies)
        return strategy(tokens, trigram, i, modified_tokens, inserted_positions)

    def _handle_noun_insertions(self, tokens: List[str], trigram: Tuple,
                              i: int, modified_tokens: List[str], 
                              inserted_positions: Set[int]) -> Tuple[List[str], Set[int], int]:
        # ... implementation of noun insertions strategy ...
        # Similar implementation as before but with better error handling
        pass

    # ... Other strategy methods ...
